"""Configurazione centralizzata, logging, rilevamento ambiente e gestione segreti."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from getpass import getpass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent


def setup_logger(name: str = "repo_agent", level: int = logging.INFO) -> logging.Logger:
    """Configura un logger strutturato per l'intero pacchetto."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s | %(name)s] %(message)s"))
        log.addHandler(handler)
    log.setLevel(level)
    return log


logger = setup_logger()


def _init_env() -> None:
    """Carica models.env (se c'è) e poi il primo .env trovato: cartella del codice, parent o cwd."""
    models_env = BASE_DIR / "models.env"
    if models_env.is_file():
        load_dotenv(dotenv_path=models_env, override=True, encoding="utf-8-sig")
    candidates = [
        BASE_DIR / ".env",
        BASE_DIR.parent / ".env",
        Path.cwd() / ".env",
        BASE_DIR / ".env.txt",         # Fallback se Windows ha aggiunto .txt
        BASE_DIR.parent / ".env.txt",
    ]
    for env_path in candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8-sig")
            found_keys = [k for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "CLOUD_API_KEY") if os.environ.get(k)]
            logger.info("File env caricato da: %s (chiavi trovate: %s)", env_path, ", ".join(found_keys) or "NESSUNA")
            return
    logger.warning(
        "Nessun file .env trovato! Cercato in: %s",
        ", ".join(str(p) for p in candidates[:3]),
    )


_init_env()


def load_providers_config() -> dict[str, Any]:
    """Carica la configurazione dei provider dal file providers.json usando il percorso assoluto dello script."""
    for json_path in (BASE_DIR / "providers.json", Path("providers.json")):
        if json_path.is_file():
            try:
                return json.loads(json_path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                logger.warning("Impossibile leggere %s: %s", json_path, exc)
    return {}


def detect_environment() -> str:
    """Rileva l'ambiente di esecuzione: 'kaggle', 'colab' oppure 'local'."""
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


RATE_LIMIT_TOKENS = (
    "429", "rate limit", "ratelimit", "resource_exhausted", "quota",
    "too many requests", "overloaded", "503",
)
# Segnali di quota GIORNALIERA esaurita: aspettare pochi secondi non serve a nulla
DAILY_QUOTA_TOKENS = ("per day", "daily", "tpd", "rpd", "per-day")


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in RATE_LIMIT_TOKENS)


def is_daily_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in DAILY_QUOTA_TOKENS)


def call_with_backoff(fn: Any, *args: Any, max_retries: int = 4, base_delay: float = 6.0,
                      label: str = "API", **kwargs: Any) -> Any:
    """Chiama fn ritentando con backoff esponenziale solo sugli errori di rate limit (HTTP 429)."""
    import time
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= max_retries or not is_rate_limit_error(exc) or is_daily_quota_error(exc):
                raise
            wait_s = base_delay * (2 ** attempt)
            logger.warning("Rate limit %s (tentativo %d/%d). Attendo %.1fs...", label, attempt + 1, max_retries, wait_s)
            time.sleep(wait_s)


def default_work_dir() -> Path:
    """Restituisce la directory di lavoro predefinita in base all'ambiente."""
    env = detect_environment()
    if env == "kaggle":
        return Path("/kaggle/working/repo_agent_work")
    if env == "colab":
        return Path("/content/repo_agent_work")
    return (BASE_DIR / "repo_agent_work").resolve()


def get_secret(name: str, prompt_if_missing: bool = True) -> str | None:
    """Cerca un segreto in: variabili d'ambiente, Colab Secrets, Kaggle Secrets o input manuale."""
    val = os.environ.get(name, "").strip().strip("\"'")
    if val:
        return val
    try:
        from google.colab import userdata  # type: ignore
        if val := userdata.get(name):
            return val.strip()
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        if val := UserSecretsClient().get_secret(name):
            return val.strip()
    except Exception:
        pass
    if prompt_if_missing:
        value = getpass(f"Inserisci {name} (non verrà mostrata): ").strip()
        if value:
            os.environ[name] = value
            return value
    return None

@dataclass
class Config:
    """Parametri globali per ingestione, chunking AST, indicizzazione scalabile e agente."""

    work_dir: Path = field(default_factory=default_work_dir)
    providers: dict[str, Any] = field(default_factory=load_providers_config)

    # --- Ingestione & Scalabilità Repo Grandi ---
    clone_depth: int | None = 50           # Clone leggero; None = storia intera
    blobless_clone: bool = True            # Scarica i blob solo su richiesta (ottimo per monorepo)
    max_file_bytes: int = 1_500_000        # Ignora singoli file > 1.5 MB (es. dump o asset generati)
    skip_dirs: tuple[str, ...] = (
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
        "target", "out", "bin", "obj", "vendor", "third_party", ".next", ".nuxt",
        ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints", "site-packages",
        "repo_agent_work",
    )
    skip_suffixes: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".pdf", ".zip", ".tar", ".gz",
        ".7z", ".whl", ".so", ".dll", ".exe", ".bin", ".pt", ".pth", ".ckpt", ".safetensors",
        ".onnx", ".npy", ".npz", ".pkl", ".h5", ".mp3", ".wav", ".flac", ".mp4", ".avi",
        ".lock", ".min.js", ".min.css", ".map", ".woff", ".woff2", ".ttf", ".pyc", ".class",
    )

    # --- Chunking ---
    max_chunk_chars: int = 3200
    window_lines: int = 65
    window_overlap: int = 12

    # --- Embedding / Indice ---
    embedder_type: str = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dims: int | None = None
    embed_max_chars: int = 1800
    embed_batch_size: int = 64
    auto_fast_embed_threshold_chunks: int = 12_000
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Agente & Rate Limit Free Tier ---
    # smolagents rimanda a OGNI step tutta la cronologia (prompt + tool + output precedenti):
    # il costo cresce circa col quadrato dei passi, quindi step e output dei tool vanno tenuti corti.
    max_steps: int = 10
    max_tool_output_chars: int = 1400      # tetto output singolo tool (1200-1500 per evitare picchi)
    read_default_lines: int = 45           # righe restituite da read_file di default (40-50 righe)
    repo_map_chars: int = 1000             # mappa strutturale compatta nel system prompt
    history_turns: int = 4
    history_answer_chars: int = 800        # testa+coda di ogni risposta passata rimandata al modello
    max_output_tokens: int = 4096          # tetto in uscita per chiamata (le risposte lunghe ne servono ~2-3k)
    llm_max_retries: int = 4
    llm_retry_base_delay: float = 6.0
    fallback_providers: tuple[str, ...] = ()   # alias di providers.json da provare se il principale va in 429
    llm_switch_after: int = 2              # fallimenti 429 sul provider corrente prima di passare al successivo
    llm_cooldown_s: float = 90.0           # quanto tempo evitare un provider appena andato in rate limit
    obs_budget_chars: int = 2500           # tetto budget osservazioni in memoria per rolling pruning
    use_answer_cache: bool = True          # riusa le risposte finché i file citati non cambiano
    fast_path_enabled: bool = True         # risoluzione RAG rapida a singolo step per query informative

    # --- Strumenti opzionali (spenti di default, attivabili da CLI) ---
    allow_exec: bool = False               # abilita run_command (su copia temporanea)
    allow_web: bool = False                # abilita web_lookup
    exec_max_copy_mb: int = 100            # dimensione massima della copia di sandbox
    exec_timeout: int = 30                 # secondi di default per run_command

    @property
    def repos_dir(self) -> Path:
        return self.work_dir / "repos"

    @property
    def index_dir(self) -> Path:
        return self.work_dir / "index"