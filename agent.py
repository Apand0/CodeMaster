"""Fase 5: Agente ReAct con supporto a providers.json (Gemini, Groq, Cloudflare) e `.env`."""
from __future__ import annotations

import atexit
import hashlib
import json
import re
import sys
import time
import shutil
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from smolagents import ToolCallingAgent, LiteLLMModel, OpenAIModel, ActionStep
from smolagents.monitoring import LogLevel

from chunker import Chunk
from config import Config, detect_environment, get_secret, is_daily_quota_error, is_rate_limit_error, logger
from indexer import RepoIndex
from citations import check_citations, render_panel
from ingest import clone_repo, file_sha256
from prompts import (
    CORE,
    DEFAULT_MODE,
    LEAN_SYSTEM_PROMPT_TEMPLATE,
    MODES,
    build_cli_instructions,
    build_instructions,
    parse_mode_prefix,
    route_mode,
    tool_names,
)
from qa_cache import AnswerCache, make_key
from tools import RepoContext, build_onboarding, build_tools


class ResilientModelWrapper:
    """Avvolge un modello smolagents: backoff sui rate limit (HTTP 429) e, se configurati,
    passaggio automatico a provider di riserva quando quello corrente resta bloccato."""

    def __init__(
        self,
        inner_model: Any,
        max_retries: int = 4,
        base_delay: float = 6.0,
        fallbacks: list[Any] | None = None,
        switch_after: int = 2,
        cooldown_s: float = 90.0,
    ) -> None:
        self._inner = inner_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._models: list[Any] = [inner_model, *(fallbacks or [])]
        self.last_model_id = self._label(inner_model)
        self.last_provider = self.last_model_id
        self._switch_after = max(1, switch_after)
        self._cooldown_s = cooldown_s
        self._cool_until: dict[int, float] = {}   # indice modello -> istante fino a cui evitarlo
        self.rate_limit_hits = 0                  # quante volte un provider ha risposto 429 (per la telemetria)

    def __getattr__(self, name: str) -> Any:
        if name in ("_inner", "_models"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    @staticmethod
    def _label(model: Any) -> str:
        return str(getattr(model, "model_id", type(model).__name__))

    def _order(self) -> list[int]:
        """Indici da provare: prima i provider non in pausa, poi (se servono) gli altri."""
        now = time.time()
        ready = [i for i in range(len(self._models)) if self._cool_until.get(i, 0) <= now]
        return ready + [i for i in range(len(self._models)) if i not in ready]

    def _with_retry(self, method: str, *args: Any, **kwargs: Any) -> Any:
        order = self._order()
        last_exc: Exception | None = None
        for pos, idx in enumerate(order):
            model = self._models[idx]
            fn = getattr(model, method, None) or model
            is_last = pos == len(order) - 1
            # provider di riserva ancora disponibile: pochi tentativi e poi si cambia, senza attese da 90s
            tries = self._max_retries if is_last else self._switch_after - 1
            for attempt in range(tries + 1):
                try:
                    result = fn(*args, **kwargs)
                    self.last_model_id = self._label(model)
                    self.last_provider = self.last_model_id
                    return result
                except Exception as exc:
                    if not is_rate_limit_error(exc):
                        raise
                    last_exc = exc
                    self.rate_limit_hits += 1
                    daily = is_daily_quota_error(exc)
                    if daily or attempt >= tries:
                        if is_last:
                            raise
                        self._cool_until[idx] = time.time() + (3600 if daily else self._cooldown_s)
                        logger.warning(
                            "Provider %s in rate limit%s: passo a %s.",
                            self._label(model), " (quota giornaliera)" if daily else "",
                            self._label(self._models[order[pos + 1]]),
                        )
                        break
                    wait_s = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit API su %s (tentativo %d/%d). Attendo %.1fs...",
                        self._label(model), attempt + 1, tries, wait_s,
                    )
                    time.sleep(wait_s)
        raise last_exc if last_exc else RuntimeError("Nessun modello disponibile")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._with_retry("generate", *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate(*args, **kwargs)


def _make_raw_model(
    provider: str, model_id: str | None, api_key: str | None, cfg: Config, **kw: Any
) -> Any:
    """Istanzia UN modello (senza wrapper) da un alias di providers.json o da 'gemini'/'groq'."""
    providers_map = cfg.providers

    # Se viene passato un alias presente in providers.json (es. "groq_120b", "cloudfare_glm-5.3", ecc.)
    if provider in providers_map:
        p_info = providers_map[provider]
        if p_info.get("type") == "cli":
            raise ValueError(f"'{provider}' è una CLI locale e non può essere creato come modello API.")
        m_id = model_id or p_info["model_id"]
        env_var = p_info.get("env_key")  # opzionale: alcuni provider free non hanno chiave
        # prompt_if_missing=False: non chiedere la chiave con getpass se non serve
        key = api_key or (get_secret(env_var, prompt_if_missing=False) if env_var else None)

        # Provider generico compatibile OpenAI (es. Kilo): basta api_base in providers.json
        if p_info.get("api_base"):
            return OpenAIModel(
                model_id=m_id,
                api_base=kw.pop("api_base", p_info["api_base"]),
                api_key=key or "not-needed",
                **kw,
            )

        # Gestione specifica per Cloudflare Workers AI (compatibile OpenAI)
        if "cloudfare" in provider or "cloudflare" in provider or env_var == "CLOUD_API_KEY":
            account_id = get_secret("CLOUD_ACCOUNT_ID", prompt_if_missing=False)
            api_base = kw.pop("api_base", f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1" if account_id else "https://api.cloudflare.com/client/v4/accounts/me/ai/v1")
            return OpenAIModel(
                model_id=m_id,
                api_base=api_base,
                api_key=key or "not-needed",
                **kw,
            )

        # Per Gemini o Groq gestiti via LiteLLM
        litellm_model = m_id
        if provider.startswith("gemini") and not litellm_model.startswith("gemini/"):
            litellm_model = "gemini/" + litellm_model
        elif provider.startswith("groq") and not litellm_model.startswith("groq/"):
            litellm_model = "groq/" + litellm_model
        return LiteLLMModel(model_id=litellm_model, api_key=key, **kw)

    # Fallback per macro-famiglie o provider standard
    prov_lower = provider.lower()
    if prov_lower in ("gemini", "groq"):
        env_var = "GEMINI_API_KEY" if prov_lower == "gemini" else "GROQ_API_KEY"
        def_m = "gemini/gemini-2.5-flash" if prov_lower == "gemini" else "groq/llama-3.3-70b-versatile"
        return LiteLLMModel(
            model_id=model_id or def_m,
            api_key=api_key or get_secret(env_var),
            **kw,
        )

    raise ValueError(f"Provider o alias '{provider}' non trovato in providers.json né tra i provider supportati.")


def _has_credentials(provider: str, cfg: Config) -> bool:
    """True se il provider può essere usato senza chiedere nulla all'utente (chiave presente o non richiesta)."""
    info = cfg.providers.get(provider)
    if info and info.get("type") == "cli":
        return bool(shutil.which(str(info.get("command", ""))))
    if info is None:
        env_var = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}.get(provider.lower())
        return bool(env_var and get_secret(env_var, prompt_if_missing=False))
    env_var = info.get("env_key")
    return not env_var or bool(get_secret(env_var, prompt_if_missing=False))


def build_model(
    provider: str = "gemini_3.8",
    model_id: str | None = None,
    api_key: str | None = None,
    cfg: Config | None = None,
    fallbacks: list[str] | tuple[str, ...] | None = None,
    **kw: Any,
) -> Any:
    """Istanzia il modello LLM (providers.json + chiavi dal `.env`), con retry e provider di riserva."""
    cfg = cfg or Config()
    # Tetto ai token in uscita: senza, un free tier può bruciare la quota con risposte/tool call enormi
    kw.setdefault("max_tokens", cfg.max_output_tokens)

    fallback_aliases = list(fallbacks if fallbacks is not None else cfg.fallback_providers)
    cli_fallbacks = [a for a in fallback_aliases if cfg.providers.get(a, {}).get("type") == "cli"]
    if cli_fallbacks:
        raise ValueError(f"--fallback accetta solo provider API; alias CLI: {', '.join(cli_fallbacks)}")
    primary = _make_raw_model(provider, model_id, api_key, cfg, **dict(kw))

    backups: list[Any] = []
    for alias in fallback_aliases:
        if alias == provider:
            continue
        if not _has_credentials(alias, cfg):
            logger.warning("Provider di riserva '%s' saltato: chiave API non trovata.", alias)
            continue
        try:
            backups.append(_make_raw_model(alias, None, None, cfg, **dict(kw)))
        except Exception as exc:
            logger.warning("Provider di riserva '%s' saltato: %s", alias, exc)
    if backups:
        logger.info("Provider di riserva attivi: %s", ", ".join(getattr(m, "model_id", "?") for m in backups))

    return ResilientModelWrapper(
        primary,
        cfg.llm_max_retries,
        cfg.llm_retry_base_delay,
        fallbacks=backups,
        switch_after=cfg.llm_switch_after,
        cooldown_s=cfg.llm_cooldown_s,
    )


class StepProgress:
    """Una sola riga di progresso che si aggiorna sul posto, senza andare a capo."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.stream = sys.stderr
        self.live = self.stream.isatty()
        self.reset()

    def reset(self) -> None:
        self.t0 = time.time()
        self.step = 0
        self.in_tok = 0
        self.out_tok = 0

    def __call__(self, step: ActionStep, agent: Any = None) -> None:
        self.step = step.step_number
        usage = getattr(step, "token_usage", None)
        if usage is not None:
            try:
                self.in_tok += int(getattr(usage, "input_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                self.out_tok += int(getattr(usage, "output_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
        self._write(f"Step {self.step}/{self.max_steps}")

    def _line(self, label: str) -> str:
        return (
            f"{label} | in: {self.in_tok:,} | out: {self.out_tok:,} "
            f"| {time.time() - self.t0:.1f}s"
        )

    def _fit(self, text: str) -> str:
        width = shutil.get_terminal_size((80, 20)).columns - 1
        return text if len(text) <= width else text[: width - 1] + "…"

    def _write(self, label: str) -> None:
        if not self.live:
            return
        self.stream.write("\r\x1b[2K" + self._fit(self._line(label)))
        self.stream.flush()

    def finish(self) -> None:
        if self.live:
            self.stream.write("\r\x1b[2K")
        self.stream.write(self._fit(self._line(f"Completato in {self.step} step")) + "\n")
        self.stream.flush()

# Pruning rolling della cronologia: solo l'ultimo step (k) mantiene l'osservazione integrale;
# tutti gli step precedenti (k-1, k-2, ...) sono ridotti a un riepilogo compatto su qualsiasi strumento (R2).
_SHRUNK = "[riepilogo compatto]"


def compact_summary(tool: str, obs: str, max_chars: int = 220) -> str:
    """Riduce l'output di qualsiasi strumento a un riepilogo compatto (firme, percorsi, intestazioni)."""
    if not obs:
        return ""
    if obs.startswith(_SHRUNK):
        return obs

    items: list[str] = []
    # 1. Chunk headers: [percorso:inizio-fine]
    chunk_heads = re.findall(r"^\[[^\]\n]+:\d+-\d+\][^\n]*", obs, re.M)
    if chunk_heads:
        items.extend(h.strip()[:100] for h in chunk_heads[:4])

    # 2. Firme di funzioni/metodi/classi (con o senza prefisso di riga '   45| ')
    signatures = re.findall(r"^\s*(?:\d+\s*\|\s*)?(?:firma:\s*|def\s+|class\s+)[^\n]+", obs, re.M)
    if signatures:
        for s in signatures[:4]:
            clean_s = re.sub(r"^\s*\d+\s*\|\s*", "", s).strip()
            if clean_s and clean_s not in items:
                items.append(clean_s[:100])

    # 3. Intestazioni read_file: percorso (righe X-Y di Z)
    read_heads = re.findall(r"^[^\n]+\(righe \d+-\d+ di \d+\)", obs, re.M)
    if read_heads:
        items.extend(r.strip()[:100] for r in read_heads[:2])

    # 4. Righe AST outline: • righe X-Y | [tipo] qualname
    ast_lines = re.findall(r"•\s*righe\s*\d+-\d+\s*\|\s*\[[^\]]+\]\s*[^\n]+", obs, re.M)
    if ast_lines:
        items.extend(a.strip()[:100] for a in ast_lines[:4])

    # 5. Corrispondenze grep/riferimenti: file:riga: testo
    ref_lines = re.findall(r"^[\w./@+-]+\.[A-Za-z0-9]{1,6}:\d+:[^\n]*", obs, re.M)
    if ref_lines:
        items.extend(g.strip()[:100] for g in ref_lines[:4])

    # 6. Fallback: prime righe non vuote
    if not items:
        lines = [ln.strip() for ln in obs.splitlines() if ln.strip()]
        items = lines[:3]

    summary_text = " | ".join(items) if items else obs[:80]
    prefix = f"{_SHRUNK} {tool} ({len(obs)} car): "
    allowed = max(40, max_chars - len(prefix))
    if len(summary_text) > allowed:
        summary_text = summary_text[: allowed - 3] + "..."
    return prefix + summary_text


# Strumenti di navigazione (intermedi: output ridotti subito a riepilogo per non consumare token)
_NAV_TOOLS = frozenset({
    "search_code", "grep", "list_dir", "get_file_outline", "find_references",
    "git_log", "git_diff", "check_python", "get_module_graph", "list_dependencies",
    "get_repo_overview", "parse_traceback",
})
# Strumenti di prova (codice sorgente: lasciati integri finché la memoria totale rientra nel budget)
_EVIDENCE_TOOLS = frozenset({
    "read_file", "find_definition", "locate_code_snippet", "read_library_source",
})


def shrink_old_observations(
    steps: list[Any], budget: int = 4000, current_step: ActionStep | None = None
) -> int:
    """Applica il pruning rolling selettivo della cronologia interna:
    1. I tool di navigazione (search_code, grep, list_dir...) nei passi precedenti (k-1, k-2...)
       vengono ridotti immediatamente a un riepilogo compatto.
    2. I tool di prova (read_file, find_definition...) restano integri nei passi storici
       finché la memoria totale non supera `budget` (ridotti solo dal più vecchio se necessario).
    3. L'osservazione dell'ultimo step corrente (k) non viene mai alterata (il modello deve ancora leggerla).
    Ritorna i caratteri liberati."""
    acts: list[ActionStep] = [s for s in steps if isinstance(s, ActionStep) and s.observations]
    if current_step is not None and isinstance(current_step, ActionStep) and current_step.observations:
        if current_step not in acts:
            acts.append(current_step)

    if len(acts) <= 1:
        return 0

    freed = 0
    # 1. Pruning rolling immediato dei soli tool di navigazione negli step precedenti (acts[:-1])
    for s in acts[:-1]:
        obs = s.observations
        if obs.startswith(_SHRUNK):
            continue
        calls = s.tool_calls or []
        tname = calls[0].name if calls else "tool"
        if tname in _NAV_TOOLS:
            shrunk = compact_summary(tname, obs)
            if len(shrunk) < len(obs):
                diff = len(obs) - len(shrunk)
                freed += diff
                s.observations = shrunk

    # 2. Controllo del budget totale: se supera budget, riduci i tool di prova dal più vecchio
    if budget > 0:
        total = sum(len(s.observations) for s in acts)
        if total > budget:
            for s in acts[:-1]:
                if total <= budget:
                    break
                obs = s.observations
                if obs.startswith(_SHRUNK):
                    continue
                calls = s.tool_calls or []
                tname = calls[0].name if calls else "tool"
                shrunk = compact_summary(tname, obs)
                if len(shrunk) < len(obs):
                    diff = len(obs) - len(shrunk)
                    freed += diff
                    total -= diff
                    s.observations = shrunk

            # Se ancora sopra budget, applica ultra-compressione sui passi storici già riassunti (mai sull'ultimo)
            if total > budget:
                for s in acts[:-1]:
                    if total <= budget:
                        break
                    if len(s.observations) > 70:
                        ultra = s.observations[:67] + "..."
                        diff = len(s.observations) - len(ultra)
                        freed += diff
                        total -= diff
                        s.observations = ultra

    return freed


@dataclass
class AskStats:
    """Telemetria dell'ultima domanda (mostrata nella UI e scritta in telemetry.jsonl)."""

    mode: str = ""
    steps: int = 0
    in_tok: int = 0
    out_tok: int = 0
    secs: float = 0.0
    retries: int = 0
    cache_hit: bool = False
    bad_citations: list[str] = field(default_factory=list)
    backend: str = "api"
    provider: str = ""
    model: str = ""
    fallback: bool = False

    def line(self) -> str:
        label = f"{self.backend}:{self.provider}" if self.provider else self.backend
        if self.fallback:
            label += " (fallback)"
        if self.cache_hit:
            return f"⚡ cache · {label}: 0 token"
        return f"{label} · {self.steps} step · in {self.in_tok:,} · out {self.out_tok:,} · {self.secs:.1f}s · {self.retries} retry"


_ML_RE = re.compile(r"\b(torch|tensorflow|jax|keras|sklearn|transformers|diffusers)\b")


def _has_ml(index: RepoIndex) -> bool:
    """True se almeno un file importa una libreria ML (serve solo per includere il blocco ML)."""
    return any(_ML_RE.search(" ".join(cs[0].imports)) for cs in index.by_file.values() if cs)


_TOOL_HINTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"\b(commit|diff|git|cronologia|storia)\b", re.I), ("git_log", "git_diff")),
    (re.compile(r"\b(traceback|stack ?trace|eccezion\w*)\b", re.I), ("parse_traceback",)),
    (re.compile(r"\b(import\w*|dipendenz\w*|moduli|requirements|pacchetti)\b", re.I), ("get_module_graph", "list_dependencies")),
    (re.compile(r"\b(punto di ingresso|entry point|panoramica|a cosa serve|onboarding)\b", re.I), ("get_repo_overview",)),
    (re.compile(r"\b(chiamat\w*|call[ _]graph|chi chiama)\b", re.I), ("get_call_graph",)),
]


class RepoAgent:
    def __init__(self, index: RepoIndex, model: Any = None, cfg: Config | None = None, *,
                 cli_backend: Any = None, provider: str = "", fallback_model: Any = None,
                 fallback_provider: str = "") -> None:
        self.index = index
        self.cfg = cfg or index.cfg
        self.progress = StepProgress(self.cfg.max_steps)
        self.mode = DEFAULT_MODE
        self.pinned = False  # True = modalità forzata dall'utente: il router automatico non interviene
        self._has_ml = _has_ml(index)
        self._stats = index.stats()
        self._repo_map = index.build_repo_map(self.cfg.repo_map_chars)
        self._all_tools = {t.name: t for t in build_tools(index)}   # istanze create UNA volta (run_command ha stato)
        self._ctx = RepoContext(index)
        self._store_dir = RepoIndex.index_path(self.cfg, index.repo_root)
        self.cache = AnswerCache(self._store_dir / "qa_cache.json")
        self.last = AskStats()
        self.cli_backend = cli_backend
        self.provider = provider or str(getattr(model, "model_id", ""))
        self.fallback_provider = fallback_provider or str(getattr(fallback_model, "model_id", ""))
        self.backend = "cli" if cli_backend is not None else "api"
        self.agent = self._new_api_agent(model) if model is not None else None
        self.fallback_agent = self._new_api_agent(fallback_model) if fallback_model is not None else None
        self._mcp_server: Any = None
        self._ask_lock = threading.RLock()
        self._response_backend = self.backend
        self._response_provider = self.provider
        self._response_model = str(getattr(cli_backend, "model_id", "") or getattr(model, "model_id", "") or self.provider)
        self._response_fallback = False
        self._cli_error = ""
        self._cli_cool_until: float = 0.0
        atexit.register(self.close)

    def _new_api_agent(self, model: Any) -> ToolCallingAgent:
        base = ToolCallingAgent(tools=[], model=model)
        templates = dict(base.prompt_templates)
        templates["system_prompt"] = LEAN_SYSTEM_PROMPT_TEMPLATE
        return ToolCallingAgent(
            tools=self._tools_for(self.mode), model=model, instructions=self._instructions(self.mode),
            max_steps=self.cfg.max_steps, verbosity_level=LogLevel.OFF,
            step_callbacks={ActionStep: [self.progress, self._shrink_memory]}, prompt_templates=templates,
        )

    def close(self) -> None:
        if self._mcp_server is not None:
            self._mcp_server.close()

    def _tools_for(self, mode: str, question: str | None = None) -> list[Any]:
        """Solo gli strumenti utili alla modalità (descrizione e schema di ogni tool costano token a OGNI step).
        Gli strumenti opzionali (web_lookup, run_command) sono inclusi solo se esplicitamente abilitati (R3).
        Include inoltre tool addizionali mirati se la domanda corrisponde a specifici _TOOL_HINTS."""
        optional: set[str] = set()
        if self.cfg.allow_web:
            optional.add("web_lookup")
        if self.cfg.allow_exec:
            optional.add("run_command")
        wanted = set(tool_names(mode)) | optional
        if question:
            for rx, thints in _TOOL_HINTS:
                if rx.search(question):
                    wanted.update(thints)
        return [t for name, t in self._all_tools.items() if name in wanted]

    def _instructions(self, mode: str) -> str:
        return build_instructions(
            mode, self.index.repo_root.name, self._stats, self._repo_map, ml=self._has_ml
        )

    def _cli_instructions(self, mode: str) -> str:
        return build_cli_instructions(
            mode, self.index.repo_root.name, self._stats, self._repo_map, ml=self._has_ml
        )

    def set_mode(self, mode: str, pin: bool = False) -> None:
        """Cambia il set di istruzioni e rigenera il system prompt di smolagents."""
        if mode not in MODES:
            raise ValueError(f"Modalità sconosciuta: {mode}")
        if pin:
            self.pinned = True
        if mode == self.mode:
            return
        self.mode = mode
        for agent in (self.agent, self.fallback_agent):
            if agent is None:
                continue
            agent.instructions = self._instructions(mode)
            final = agent.tools.get("final_answer")
            agent.tools = {t.name: t for t in self._tools_for(mode)}
            if final is not None:
                agent.tools["final_answer"] = final
            self._refresh_system_prompt(agent)

    def _refresh_system_prompt(self, agent: Any = None) -> None:
        """system_prompt è read-only: si aggiorna invece il passo di sistema in memoria."""
        from smolagents.memory import SystemPromptStep

        agent = agent or self.agent or self.fallback_agent
        if agent is None:
            return
        agent.memory.system_prompt = SystemPromptStep(
            system_prompt=agent.initialize_system_prompt()
        )

    def unpin(self) -> None:
        self.pinned = False
        self.set_mode(DEFAULT_MODE)

    def _shrink_memory(self, step: ActionStep, agent: Any = None) -> None:
        agent = agent or self.agent or self.fallback_agent
        if agent is None:
            return
        freed = shrink_old_observations(
            agent.memory.steps,
            budget=self.cfg.obs_budget_chars,
            current_step=step,
        )
        if freed:
            logger.debug("Memoria dell'agente: liberati %d caratteri di output vecchi.", freed)

    def _sha(self, rel: str) -> str:
        try:
            return file_sha256(self.index.repo_root / rel)
        except OSError:
            return ""

    def _cache_key(self, question: str, history: list[tuple[str, str]] | None) -> str | None:
        if not self.cfg.use_answer_cache:
            return None
        sig = hashlib.sha1((CORE + MODES[self.mode].body).encode("utf-8")).hexdigest()[:8]
        prev = [q for q, _ in (history or [])[-self.cfg.history_turns :]]
        model_id = (self.provider if self.cli_backend is not None else
                    str(getattr(self.agent.model, "model_id", "")))
        return make_key(question, self.mode, model_id, sig, prev)

    def _log_telemetry(self, question: str) -> None:
        try:
            m_id = self.last.model or (self.provider if self.cli_backend is not None else
                                       str(getattr(self.agent.model, "model_id", "")))
            data = asdict(self.last)
            for k, v in list(data.items()):
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    data[k] = str(v)
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "q": question[:80],
                "model": m_id,
                "backend": self.last.backend,
                "provider": self.last.provider,
                **data,
            }
            with open(self._store_dir / "telemetry.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("Telemetria non scritta: %s", exc)

    def _local_mcp(self, active_tools: list[Any]) -> tuple[str, str]:
        if self._mcp_server is None:
            from mcp_server import LocalMCPServer
            self._mcp_server = LocalMCPServer()
        self._mcp_server.set_tools(active_tools)
        return self._mcp_server.start()

    def onboarding(self) -> str:
        """Guida iniziale alla repo (0 token LLM)."""
        return build_onboarding(self._ctx, self._all_tools["get_module_graph"])

    def citation_panel(self, answer: str) -> str:
        """Blocchi <details> con il codice delle citazioni verificate nella risposta (0 token LLM)."""
        report = check_citations(answer, self.index.files, self._ctx.file_text)
        return render_panel(report, self._ctx.file_text)

    def _shorten(self, answer: str) -> str:
        """Accorcia una risposta passata tenendo l'inizio E la fine (dove stanno citazioni e 'Limiti')."""
        n = self.cfg.history_answer_chars
        if len(answer) <= n:
            return answer
        half = n // 2
        return answer[:half] + "\n[...]\n" + answer[-half:]

    def _compose_task(self, question: str, history: list[tuple[str, str]] | None) -> str:
        if not history:
            return question
        turns = history[-self.cfg.history_turns :]
        past = "\n".join(f"Utente: {q}\nAssistente: {self._shorten(a)}" for q, a in turns)
        return f"Conversazione precedente (contesto):\n{past}\n\nNuova domanda dell'utente:\n{question}"

    _COMPLEX_QUERY_RE = re.compile(
        r"\b(bug|error\w*|sbagli\w*|non funziona|crash\w*|eccezion\w*|traceback|"
        r"race condition|leak|vulnerabil\w*|difett\w*|off-by-one|debug\w*|"
        r"refactor\w*|miglior\w*|ottimizz\w*|architettur\w*|riscriv\w*|modific\w*|"
        r"patch\w*|diff|commit|git_diff|perch[eé] (?:fallisce|non va)|"
        r"esegui|run_command|terminale|"
        r"chi chiama|dove [eè] usat\w*|chiamant\w*|call[ _]graph|impatto|dipendenz\w*|"
        r"references|riferiment\w*|storia|cronologia)\b",
        re.IGNORECASE,
    )

    _AMBIGUOUS_NAMES = frozenset({
        "main", "run", "load", "forward", "step", "get", "set", "init", "update", "reset",
        "save", "parse", "process", "call", "start", "stop", "test",
    })

    def _find_symbol_chunks(self, question: str) -> tuple[list[Chunk], str]:
        """Recupera chunk di definizione esatta per simboli citati nella domanda (0 token API).
        Se il simbolo è comune o ha > 3 definizioni ambigue, delega al ReAct loop."""
        for match in re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", question):
            short = match.split(".")[-1].lower()
            if short in self._AMBIGUOUS_NAMES:
                continue
            found = self.index.find_symbol(match)
            if found and len(found) <= 3:
                return found[:3], f"Simbolo AST: {match}"
        for dotted in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]+)+\b", question):
            found = self.index.find_symbol(dotted)
            if found and len(found) <= 3:
                return found[:3], f"Simbolo AST: {dotted}"
        words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", question)
        stop = {"file", "funzione", "classe", "metodo", "codice", "repo", "cosa", "come", "dove", "quali", "quale"}
        for w in words:
            if w.lower() in stop or w.lower() in self._AMBIGUOUS_NAMES:
                continue
            if w in self.index.symbols:
                found = self.index.find_symbol(w)
                if found and len(found) <= 3:
                    return found[:3], f"Simbolo AST: {w}"
        return [], ""

    def _find_file_overview(self, question: str) -> tuple[list[Chunk], str]:
        """Recupera outline e chunk iniziali se la query chiede la panoramica di un file specifico."""
        q_low = question.lower()
        is_overview = any(
            t in q_low
            for t in ("panoramica", "struttura", "spiegami", "overview", "outline", "file", "cosa fa", "cosa contiene")
        )
        if not is_overview:
            return [], ""
        for f in self.index.files:
            fname = f.split("/")[-1]
            if f in question or re.search(rf"(?<![A-Za-z0-9_]){re.escape(fname)}(?![A-Za-z0-9_])", question, re.I):
                outline = self.index.get_file_outline(f)
                file_chunks = self.index.by_file.get(f, [])[:3]
                return file_chunks, outline[:1000]
        return [], ""

    def _find_hybrid_chunks(self, question: str) -> tuple[list[Chunk], str]:
        """Ricerca ibrida (BM25 + embeddings locali) ad alta confidenza.
        Richiede best_score >= 0.028 per garantire accordo dual-retriever (BM25 + semantico)."""
        results = self.index.search(question, k=4)
        if not results or not isinstance(results, list):
            return [], ""
        try:
            best_score = float(results[0][1])
        except (TypeError, ValueError, IndexError):
            return [], ""
        if best_score < 0.028:
            return [], ""
        selected: list[Chunk] = []
        for item in results[:3]:
            try:
                c, sc = item
                if float(sc) >= 0.018:
                    selected.append(c)
            except (TypeError, ValueError, IndexError):
                continue
        return selected, ""

    def _numbered_chunk(self, c: Chunk, max_lines: int = 40) -> str:
        """Restituisce il codice del chunk numerato sulle righe REALI del file sorgente."""
        try:
            raw = self._ctx.file_text(c.path)
            lines = raw.splitlines()
        except Exception:
            lines = c.text.splitlines()
        start = max(1, c.start)
        end = min(c.end, start + max_lines - 1, len(lines))
        body = "\n".join(f"{i:>5}| {lines[i-1]}" for i in range(start, end + 1))
        if end < c.end:
            body += f"\n      [... righe {end+1}-{c.end} non mostrate]"
        return body

    def _try_fast_path(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> tuple[str, list[str]] | None:
        """Risoluzione RAG a step singolo per query informative, simboli o file.
        Fallback trasparente (ritorna None) se la query è complessa o le citazioni falliscono (R1)."""
        if not getattr(self.cfg, "fast_path_enabled", True):
            return None
        if self.mode in ("bugs", "design"):
            return None
        if self._COMPLEX_QUERY_RE.search(question):
            return None

        chunks: list[Chunk] = []
        extra_ctx = ""
        # 1. Simbolo singolo
        sym_chunks, sym_ctx = self._find_symbol_chunks(question)
        if sym_chunks:
            chunks, extra_ctx = sym_chunks, sym_ctx
        else:
            # 2. Panoramica file
            file_chunks, file_ctx = self._find_file_overview(question)
            if file_chunks or file_ctx:
                chunks, extra_ctx = file_chunks, file_ctx
            else:
                # 3. Ricerca ibrida locale ad alta confidenza
                hyb_chunks, hyb_ctx = self._find_hybrid_chunks(question)
                if hyb_chunks:
                    chunks, extra_ctx = hyb_chunks, hyb_ctx

        if not chunks and not extra_ctx:
            return None

        repo_name = self.index.repo_root.name
        sample_cite = f"{chunks[0].path}:{chunks[0].start}-{chunks[0].end}" if chunks else "file.py:1-20"
        system_prompt = (
            f'Sei una guida esperta di codice per la repo "{repo_name}". '
            "Rispondi alla domanda usando ESCLUSIVAMENTE il contesto di codice fornito.\n\n"
            "REGOLE TASSATIVE:\n"
            "1. CITAZIONI OBBLIGATORIE: ogni affermazione tecnica sul codice DEVE contenere la citazione "
            f"esatta nel formato `percorso/file.ext:riga_inizio-riga_fine` (es. `{sample_cite}`).\n"
            "2. VERITÀ: Basati unicamente sulle righe presenti nel contesto. Non inventare percorsi o righe.\n"
            "3. FORMATO:\n"
            "- 'In parole semplici': 1-3 frasi chiare ed essenziali.\n"
            "- 'Nel codice': spiegazione tecnica dettagliata con citazioni esatte `file:righe` a fine frase.\n"
            "- 'Limiti': dichiara cosa non è visibile nel contesto fornito.\n"
            "4. USCITA DI SICUREZZA: se il contesto fornito non è sufficiente per una risposta completa "
            "o la domanda richiede analisi multi-file/chiamanti/modifiche non presenti nel testo, "
            "rispondi ESATTAMENTE E SOLO: NEED_TOOLS\n"
        )

        blocks: list[str] = []
        if extra_ctx:
            blocks.append(f"### Dati strutturali:\n{extra_ctx}")
        for c in chunks:
            h = f"[{c.path}:{c.start}-{c.end}] {c.kind} {c.qualname}"
            if c.signature:
                h += f"\n  firma: {c.signature}"
            if c.docstring:
                h += f"\n  doc: {c.docstring.splitlines()[0][:200]}"
            numbered = self._numbered_chunk(c, max_lines=40)
            blocks.append(f"{h}\n{numbered}")

        history_prefix = ""
        if history:
            turns = history[-2:]
            past = "\n".join(f"Utente: {q}\nAssistente: {self._shorten(a)}" for q, a in turns)
            history_prefix = f"Conversazione precedente (contesto):\n{past}\n\n"

        user_content = (
            f"{history_prefix}Domanda dell'utente:\n{question}\n\n"
            f"Contesto di codice dalla repository:\n" + "\n\n---\n\n".join(blocks)
        )

        t0 = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = None
        cli_result = None
        now = time.time()
        cli_available = (self.cli_backend is not None and
                         now >= getattr(self, "_cli_cool_until", 0.0) and
                         not self._cli_error)
        if cli_available:
            try:
                cli_result = self.cli_backend.complete(system_prompt, user_content)
            except Exception as exc:
                from cli_backend import CliResult
                cli_result = CliResult(error=f"Invocazione CLI fallita: {type(exc).__name__}: {str(exc)[:400]}")
            if not cli_result.ok:
                self._cli_error = cli_result.error
                daily = is_daily_quota_error(Exception(cli_result.error))
                cooldown = float(getattr(getattr(self, "cfg", None), "llm_cooldown_s", 90.0) or 90.0)
                self._cli_cool_until = time.time() + (3600.0 if daily else cooldown)
                if self.fallback_agent is None:
                    self.last.backend = "cli"
                    self.last.provider = self.provider
                    self.last.model = self.provider
                    self.last.secs = time.time() - t0
                    self._log_telemetry(question)
                    return f"Errore CLI {self.provider}: {cli_result.error}", [f"modo={self.mode}", "cli_error"]
                try:
                    resp = self.fallback_agent.model.generate(messages=messages)
                    self._response_backend = "api"
                    self._response_provider = self.fallback_provider
                    self._response_model = str(getattr(self.fallback_agent.model, "model_id", self.fallback_provider))
                    self._response_fallback = True
                except Exception as exc:
                    logger.warning("Fast-Path CLI/API fallito (%s): passo al ciclo agentico API.", exc)
                    return None
            else:
                answer = cli_result.text.strip()
                self._response_backend = "cli"
                self._response_provider = self.provider
                self._response_model = str(getattr(self.cli_backend, "model_id", "") or self.provider)
                self._response_fallback = False
        elif self.cli_backend is not None:
            if self.fallback_agent is None:
                return f"CLI {self.provider} temporaneamente in cooldown.", [f"modo={self.mode}", "cli_cooldown"]
            try:
                resp = self.fallback_agent.model.generate(messages=messages)
                self._response_backend = "api"
                self._response_provider = self.fallback_provider
                self._response_model = str(getattr(self.fallback_agent.model, "model_id", self.fallback_provider))
                self._response_fallback = True
            except Exception as exc:
                logger.warning("Fast-Path generation fallita su fallback API (%s): passo al ReAct loop.", exc)
                return None
        else:
            try:
                resp = self.agent.model.generate(messages=messages)
            except Exception as exc:
                logger.warning("Fast-Path generation fallita (%s): fallback trasparente al ReAct loop.", exc)
                return None

        if resp is not None:
            usage = getattr(resp, "token_usage", None)
            in_val = getattr(usage, "input_tokens", 0) if usage else 0
            out_val = getattr(usage, "output_tokens", 0) if usage else 0
            try:
                in_tok = int(in_val or 0)
            except (TypeError, ValueError):
                in_tok = 0
            try:
                out_tok = int(out_val or 0)
            except (TypeError, ValueError):
                out_tok = 0
            answer = str(getattr(resp, "content", "") or "").strip()
            model = self.fallback_agent.model if self._response_fallback else getattr(self.agent, "model", None)
            if model is not None:
                self._response_model = str(getattr(model, "last_model_id", getattr(model, "model_id", self._response_model)))
                self._response_provider = str(getattr(model, "last_provider", self._response_provider))
        else:
            in_tok = int(cli_result.in_tokens or 0)
            out_tok = int(cli_result.out_tokens or 0)
        self._last_fast_tokens = (in_tok, out_tok)

        if not answer or "NEED_TOOLS" in answer:
            logger.info(
                "Fast-Path non applicabile ('%s'): fallback trasparente al ReAct loop.",
                "NEED_TOOLS" if "NEED_TOOLS" in answer else "risposta vuota",
            )
            return None

        report = check_citations(answer, self.index.files, self._ctx.file_text)
        if not report.valid or report.problems:
            logger.info(
                "Fast-Path citazioni non convalidate (valid=%d, problems=%s): fallback trasparente al ReAct loop.",
                len(report.valid),
                report.problems,
            )
            return None

        st = self.last
        st.steps = 1
        st.in_tok = in_tok
        st.out_tok = out_tok
        st.secs = time.time() - t0
        st.retries = 0
        st.bad_citations = report.problems
        st.backend = self._response_backend
        st.provider = self._response_provider
        st.model = self._response_model
        st.fallback = self._response_fallback

        key = self._cache_key(question, history)
        if key is not None and report.valid and not report.problems:
            deps = {rel: self._sha(rel) for rel in {r for r, _, _ in report.valid}}
            if "" not in deps.values():
                self.cache.put(key, question, answer, deps, backend=st.backend, provider=st.provider,
                               model=st.model, fallback=st.fallback)
        self._log_telemetry(question)

        trace = [f"modo={self.mode}", "fast_path"]
        return answer, trace

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
        # Tool allow-lists and agent memory are per-question state; serialize shared UI requests.
        with self._ask_lock:
            return self._ask_locked(question, history)

    def _ask_locked(self, question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
        cmd, forced, question = parse_mode_prefix(question)
        if cmd == "/auto":
            self.unpin()
            if not question:
                return "Routing automatico riattivato.", []
            routed = route_mode(question)
            if routed:
                self.set_mode(routed)
        elif forced:
            self.set_mode(forced, pin=True)
            if not question:
                return f"Modalità fissata: {MODES[forced].label}. Usa /auto per tornare al routing automatico.", []
        elif not self.pinned:
            routed = route_mode(question)
            if routed:
                self.set_mode(routed)

        # Aggiorna dinamicamente i tool dell'agente con eventuali tool hints per la specifica domanda
        active_tools = self._tools_for(self.mode, question)
        for api_agent in (self.agent, self.fallback_agent):
            if api_agent is None:
                continue
            final_tool = api_agent.tools.get("final_answer")
            api_agent.tools = {t.name: t for t in active_tools}
            if final_tool is not None:
                api_agent.tools["final_answer"] = final_tool

        now = time.time()
        if self.cli_backend is not None and getattr(self, "_cli_cool_until", 0.0) > now:
            logger.info(
                "CLI %s in cooldown per altri %.1fs; passaggio automatico al fallback API.",
                self.provider,
                self._cli_cool_until - now,
            )
            self._cli_error = f"CLI in cooldown per altri {int(self._cli_cool_until - now)}s"
        else:
            self._cli_error = ""
        self._response_backend = self.backend
        self._response_provider = self.provider
        self._response_model = str(getattr(self.cli_backend, "model_id", "") or
                                   getattr(getattr(self.agent, "model", None), "model_id", "") or self.provider)
        self._response_fallback = False
        self.last = AskStats(
            mode=self.mode, backend=self.backend, provider=self.provider,
            model=self._response_model,
        )
        key = self._cache_key(question, history)
        if key is not None:
            record = self.cache.get_record(key, self._sha)
            if record is not None:
                self.last.cache_hit = True
                self.last.backend = str(record.get("backend", self.backend))
                self.last.provider = str(record.get("provider", self.provider))
                self.last.model = str(record.get("model", self.last.model))
                self.last.fallback = bool(record.get("fallback", False))
                self._log_telemetry(question)
                return str(record["answer"]), [f"modo={self.mode}", "cache"]

        self._last_fast_tokens = (0, 0)
        # R1: Fast-Path RAG a step singolo per query informative / consultazione simboli o file
        fast_res = self._try_fast_path(question, history)
        if fast_res is not None:
            return fast_res

        self.progress.reset()
        fast_in, fast_out = getattr(self, "_last_fast_tokens", (0, 0))
        try:
            self.progress.in_tok += int(fast_in or 0)
        except (TypeError, ValueError):
            pass
        try:
            self.progress.out_tok += int(fast_out or 0)
        except (TypeError, ValueError):
            pass
        api_agent = self.fallback_agent if self._cli_error else self.agent
        cli_trace: list[str] = []
        retries0 = getattr(getattr(api_agent, "model", None), "rate_limit_hits", 0)
        cli_result = None
        from cli_backend import CliResult
        try:
            if self.cli_backend is not None and not self._cli_error:
                try:
                    url, token = self._local_mcp(active_tools)
                    cli_result = self.cli_backend.agentic(
                        self._cli_instructions(self.mode), self._compose_task(question, history), url, token,
                        [tool.name for tool in active_tools],
                    )
                except Exception as exc:
                    cli_result = CliResult(error=f"Invocazione CLI/MCP fallita: {type(exc).__name__}: {str(exc)[:400]}")
                cli_trace = self._mcp_server.take_trace() if self._mcp_server is not None else []
                if cli_result.ok:
                    if active_tools and not cli_trace:
                        rep = check_citations(cli_result.text, self.index.files, self._ctx.file_text)
                        if not rep.valid or rep.problems:
                            logger.warning(
                                "CLI %s ha risposto senza tool MCP e le citazioni non sono convalidate; fallback all'API.",
                                self.provider,
                            )
                            cli_result = CliResult(
                                error="La CLI non ha utilizzato gli strumenti MCP e le citazioni non sono convalidate.",
                                ok=False,
                            )
                if cli_result.ok:
                    answer = cli_result.text
                    self._response_backend = "cli"
                    self._response_provider = self.provider
                    self._response_model = str(getattr(self.cli_backend, "model_id", "") or self.provider)
                    self._response_fallback = False
                    self.progress.step = max(cli_result.turns, len(cli_trace))
                    self.progress.in_tok += cli_result.in_tokens
                    self.progress.out_tok += cli_result.out_tokens
                else:
                    self._cli_error = cli_result.error
                    daily = is_daily_quota_error(Exception(cli_result.error))
                    cooldown = float(getattr(getattr(self, "cfg", None), "llm_cooldown_s", 90.0) or 90.0)
                    self._cli_cool_until = time.time() + (3600.0 if daily else cooldown)
                    if self.fallback_agent is None:
                        self.progress.finish()
                        self.last.backend = "cli"
                        self.last.provider = self.provider
                        self.last.model = str(getattr(self.cli_backend, "model_id", "") or self.provider)
                        self.last.steps = len(cli_trace)
                        self.last.in_tok = self.progress.in_tok
                        self.last.out_tok = self.progress.out_tok
                        self.last.secs = time.time() - self.progress.t0
                        self._log_telemetry(question)
                        return f"Errore CLI {self.provider}: {cli_result.error}", [f"modo={self.mode}", *cli_trace, "cli_error"]
                    api_agent = self.fallback_agent
                    self._response_backend = "api"
                    self._response_provider = self.fallback_provider
                    self._response_model = str(getattr(api_agent.model, "model_id", self.fallback_provider))
                    self._response_fallback = True
                    retries0 = getattr(api_agent.model, "rate_limit_hits", 0)
                    answer = api_agent.run(self._compose_task(question, history), reset=True)
                    self._response_model = str(getattr(api_agent.model, "last_model_id", api_agent.model.model_id))
                    self._response_provider = str(getattr(api_agent.model, "last_provider", self._response_provider))
            else:
                if api_agent is None:
                    raise RuntimeError("Il provider CLI è fallito o in cooldown e non è configurato un fallback API.")
                if self.cli_backend is not None:
                    self._response_backend = "api"
                    self._response_provider = self.fallback_provider
                    self._response_model = str(getattr(api_agent.model, "model_id", self.fallback_provider))
                    self._response_fallback = True
                answer = api_agent.run(self._compose_task(question, history), reset=True)
                if self._response_backend == "api":
                    self._response_model = str(getattr(api_agent.model, "last_model_id", getattr(api_agent.model, "model_id", self.provider)))
                    self._response_provider = str(getattr(api_agent.model, "last_provider", self._response_provider))
        except Exception as exc:
            self.progress.finish()
            self.last.backend = self._response_backend
            self.last.provider = self._response_provider
            self.last.model = self._response_model
            self.last.fallback = self._response_fallback
            self.last.secs = time.time() - self.progress.t0
            self._log_telemetry(question)
            return f"Errore durante la chiamata al modello: {type(exc).__name__}: {str(exc)[:400]}", []
        self.progress.finish()
        answer = str(answer)

        st = self.last
        st.steps, st.in_tok, st.out_tok = self.progress.step, self.progress.in_tok, self.progress.out_tok
        st.secs = time.time() - self.progress.t0
        st.backend = self._response_backend
        st.provider = self._response_provider
        st.model = self._response_model
        st.fallback = self._response_fallback
        try:
            hits = int(getattr(getattr(api_agent, "model", None), "rate_limit_hits", 0) or 0)
            st.retries = max(0, hits - int(retries0 or 0))
        except (TypeError, ValueError):
            st.retries = 0
        report = check_citations(answer, self.index.files, self._ctx.file_text)
        st.bad_citations = report.problems
        completed = bool(cli_result and cli_result.ok) or any(
            getattr(s, "is_final_answer", False) for s in getattr(getattr(api_agent, "memory", None), "steps", [])
        )
        if key is not None and completed and report.valid and not report.problems:
            deps = {rel: self._sha(rel) for rel in {r for r, _, _ in report.valid}}
            if "" not in deps.values():
                self.cache.put(key, question, answer, deps, backend=st.backend, provider=st.provider,
                               model=st.model, fallback=st.fallback)
        self._log_telemetry(question)

        trace: list[str] = [f"modo={self.mode}"]
        if self._response_fallback:
            trace.append(f"fallback=api:{self._response_provider}")
        trace.extend(cli_trace)
        if api_agent is not None:
            for step in api_agent.memory.steps:
                for tc in getattr(step, "tool_calls", None) or []:
                    if tc.name == "final_answer":
                        continue
                    args = (
                        ", ".join(f"{k}={v!r}"[:60] for k, v in (tc.arguments or {}).items())
                        if isinstance(tc.arguments, dict)
                        else str(tc.arguments)[:80]
                    )
                    trace.append(f"{tc.name}({args})")
        return answer, trace


def create_agent(
    source: str,
    provider: str = "gemini_3.8",
    model_id: str | None = None,
    rebuild: bool = False,
    cfg: Config | None = None,
    embedder: Any = None,
    fallbacks: list[str] | None = None,
    **model_kw: Any,
) -> RepoAgent:
    cfg = cfg or Config()
    provider_info = cfg.providers.get(provider, {})
    fallback_aliases = list(fallbacks if fallbacks is not None else cfg.fallback_providers)
    unknown_fallbacks = [a for a in fallback_aliases if a not in cfg.providers and a.lower() not in ("gemini", "groq")]
    if unknown_fallbacks:
        raise ValueError(f"Provider di fallback sconosciuti: {', '.join(unknown_fallbacks)}")
    invalid_fallbacks = [a for a in fallback_aliases if cfg.providers.get(a, {}).get("type") == "cli"]
    if invalid_fallbacks:
        raise ValueError(f"Il fallback deve usare alias API, non CLI: {', '.join(invalid_fallbacks)}")
    if provider_info.get("type") == "cli":
        if detect_environment() != "local":
            raise RuntimeError("Le CLI LLM locali richiedono un terminale locale con binario e login disponibili; Kaggle/Colab non sono supportati.")
        from cli_backend import CliBackend
        cli_backend = CliBackend(provider, provider_info, model_id=model_id,
                                 timeout=cfg.cli_timeout, max_turns=cfg.max_steps)
        fallback_model = None
        fallback_provider = ""
        if fallback_aliases:
            fallback_provider = fallback_aliases[0]
            fallback_model = build_model(fallback_provider, cfg=cfg,
                                         fallbacks=fallback_aliases[1:], **model_kw)
        root = clone_repo(source, cfg)
        index = RepoIndex.build(root, cfg, embedder=embedder, force_full=rebuild)
        return RepoAgent(index, cfg=cfg, cli_backend=cli_backend, provider=provider,
                         fallback_model=fallback_model, fallback_provider=fallback_provider)

    root = clone_repo(source, cfg)
    index = RepoIndex.build(root, cfg, embedder=embedder, force_full=rebuild)
    model = build_model(provider, model_id, cfg=cfg, fallbacks=fallbacks, **model_kw)
    return RepoAgent(index, model, cfg, provider=provider)
