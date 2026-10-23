"""Fase 5: Agente ReAct con supporto a providers.json (Gemini, Groq, Cloudflare) e `.env`."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

from smolagents import ToolCallingAgent, LiteLLMModel, OpenAIModel, ActionStep
from smolagents.monitoring import LogLevel

from config import Config, get_secret, is_daily_quota_error, is_rate_limit_error, logger
from indexer import RepoIndex
from citations import check_citations, render_panel
from ingest import clone_repo, file_sha256
from prompts import CORE, DEFAULT_MODE, MODES, build_instructions, parse_mode_prefix, route_mode, tool_names
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
                    return fn(*args, **kwargs)
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

    primary = _make_raw_model(provider, model_id, api_key, cfg, **dict(kw))

    backups: list[Any] = []
    for alias in (fallbacks if fallbacks is not None else cfg.fallback_providers):
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
            self.in_tok += usage.input_tokens or 0
            self.out_tok += usage.output_tokens or 0
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

# Strumenti il cui output è soprattutto codice: il testo vecchio si può rileggere, l'intestazione basta a ricordare cosa è stato letto
_SHRINKABLE = frozenset({
    "read_file", "search_code", "find_definition", "locate_code_snippet", "read_library_source", "git_diff", "web_lookup",
})
_CHUNK_HEAD = re.compile(r"^\[[^\]\n]+:\d+-\d+\][^\n]*", re.M)
_SHRUNK = "[output ridotto]"
_KEEP_RECENT = 2   # gli ultimi step restano sempre integri


def _shrink(tool: str, obs: str) -> str:
    heads = [h[:160] for h in _CHUNK_HEAD.findall(obs)[:12]] or [obs.split("\n", 1)[0][:200]]
    return (
        f"{_SHRUNK} {tool}: {len(obs)} caratteri ridotti alle intestazioni. "
        "Se ti serve il testo, rilancia lo strumento con un intervallo stretto.\n" + "\n".join(heads)
    )


def shrink_old_observations(steps: list[Any], budget: int) -> int:
    """Se le observations in memoria superano `budget` caratteri, riduce le più vecchie (dal più vecchio) alle sole
    intestazioni. Smolagents rimanda tutta la cronologia a ogni step: così il costo smette di crescere.
    Ritorna i caratteri liberati."""
    if budget <= 0:
        return 0
    acts = [s for s in steps if isinstance(s, ActionStep) and s.observations]
    total = sum(len(s.observations) for s in acts)
    freed = 0
    for s in acts[:-_KEEP_RECENT]:
        if total <= budget:
            break
        calls = s.tool_calls or []
        if not calls or s.observations.startswith(_SHRUNK) or not all(tc.name in _SHRINKABLE for tc in calls):
            continue
        new = _shrink(calls[0].name, s.observations)
        if len(new) < len(s.observations):
            freed += len(s.observations) - len(new)
            total -= len(s.observations) - len(new)
            s.observations = new
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

    def line(self) -> str:
        if self.cache_hit:
            return "⚡ risposta dalla cache: 0 token"
        return f"{self.steps} step · in {self.in_tok:,} · out {self.out_tok:,} · {self.secs:.1f}s · {self.retries} retry"


_ML_RE = re.compile(r"\b(torch|tensorflow|jax|keras|sklearn|transformers|diffusers)\b")


def _has_ml(index: RepoIndex) -> bool:
    """True se almeno un file importa una libreria ML (serve solo per includere il blocco ML)."""
    return any(_ML_RE.search(" ".join(cs[0].imports)) for cs in index.by_file.values() if cs)


class RepoAgent:
    def __init__(self, index: RepoIndex, model: Any, cfg: Config | None = None) -> None:
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
        self.agent = ToolCallingAgent(
            tools=self._tools_for(self.mode),
            model=model,
            instructions=self._instructions(self.mode),
            max_steps=self.cfg.max_steps,
            verbosity_level=LogLevel.OFF,
            step_callbacks={ActionStep: [self.progress, self._shrink_memory]},
        )

    def _tools_for(self, mode: str) -> list[Any]:
        """Solo gli strumenti utili alla modalità (descrizione e schema di ogni tool costano token a OGNI step).
        Gli strumenti opzionali (web_lookup, run_command) restano sempre, se abilitati dall'utente."""
        wanted = tool_names(mode) | {"web_lookup", "run_command"}
        return [t for name, t in self._all_tools.items() if name in wanted]

    def _instructions(self, mode: str) -> str:
        return build_instructions(
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
        self.agent.instructions = self._instructions(mode)
        final = self.agent.tools.get("final_answer")
        self.agent.tools = {t.name: t for t in self._tools_for(mode)}
        if final is not None:
            self.agent.tools["final_answer"] = final
        self._refresh_system_prompt()

    def _refresh_system_prompt(self) -> None:
        """system_prompt è read-only: si aggiorna invece il passo di sistema in memoria."""
        from smolagents.memory import SystemPromptStep

        self.agent.memory.system_prompt = SystemPromptStep(
            system_prompt=self.agent.initialize_system_prompt()
        )

    def unpin(self) -> None:
        self.pinned = False
        self.set_mode(DEFAULT_MODE)

    def _shrink_memory(self, step: ActionStep, agent: Any = None) -> None:
        freed = shrink_old_observations(self.agent.memory.steps, self.cfg.obs_budget_chars)
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
        return make_key(question, self.mode, str(getattr(self.agent.model, "model_id", "")), sig, prev)

    def _log_telemetry(self, question: str) -> None:
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "q": question[:80],
            "model": str(getattr(self.agent.model, "model_id", "")), **asdict(self.last),
        }
        try:
            with open(self._store_dir / "telemetry.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug("Telemetria non scritta: %s", exc)

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

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
        cmd, forced, question = parse_mode_prefix(question)
        if cmd == "/auto":
            self.unpin()
            if not question:
                return "Routing automatico riattivato.", []
        elif forced:
            self.set_mode(forced, pin=True)
            if not question:
                return f"Modalità fissata: {MODES[forced].label}. Usa /auto per tornare al routing automatico.", []
        elif not self.pinned:
            routed = route_mode(question)
            if routed:
                self.set_mode(routed)

        self.last = AskStats(mode=self.mode)
        key = self._cache_key(question, history)
        if key is not None:
            cached = self.cache.get(key, self._sha)
            if cached is not None:
                self.last.cache_hit = True
                self._log_telemetry(question)
                return cached, [f"modo={self.mode}", "cache"]

        self.progress.reset()
        retries0 = getattr(self.agent.model, "rate_limit_hits", 0)
        try:
            answer = self.agent.run(self._compose_task(question, history), reset=True)
        except Exception as exc:
            self.progress.finish()
            return f"Errore durante la chiamata al modello: {type(exc).__name__}: {str(exc)[:400]}", []
        self.progress.finish()
        answer = str(answer)

        st = self.last
        st.steps, st.in_tok, st.out_tok = self.progress.step, self.progress.in_tok, self.progress.out_tok
        st.secs = time.time() - self.progress.t0
        st.retries = getattr(self.agent.model, "rate_limit_hits", 0) - retries0
        report = check_citations(answer, self.index.files, self._ctx.file_text)
        st.bad_citations = report.problems
        # si salva solo una risposta completa, con almeno una citazione verificata e nessuna citazione sospetta
        completed = any(getattr(s, "is_final_answer", False) for s in self.agent.memory.steps)
        if key is not None and completed and report.valid and not report.problems:
            deps = {rel: self._sha(rel) for rel in {r for r, _, _ in report.valid}}
            if "" not in deps.values():
                self.cache.put(key, question, answer, deps)
        self._log_telemetry(question)

        trace: list[str] = [f"modo={self.mode}"]
        for step in self.agent.memory.steps:
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
    root = clone_repo(source, cfg)
    index = RepoIndex.build(root, cfg, embedder=embedder, force_full=rebuild)
    model = build_model(provider, model_id, cfg=cfg, fallbacks=fallbacks, **model_kw)
    return RepoAgent(index, model, cfg)