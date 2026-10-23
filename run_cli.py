"""CLI interattiva per usare l'agente direttamente nel terminale del tuo IDE (VS Code, PyCharm, ecc.)."""
from __future__ import annotations

import argparse
import os
import re

from agent import create_agent
from config import Config, load_providers_config
from prompts import MODE_COMMANDS, MODES

HELP_BANNER = """
================================================================================
REPO STUDY AGENT - Modalità Terminale / IDE (con supporto providers.json)
Comandi rapidi disponibili in chat:
  /outline <file>   -> Mostra subito la struttura AST di un file (0 token LLM)
  /symbol <nome>    -> Trova subito la definizione di una funzione/classe (0 token LLM)
  /map              -> Mostra la mappa generale della repository
  /onboarding       -> Guida iniziale alla repo: da dove partire, file chiave (0 token LLM)
  /teoria | /pratica | /bug | /design [domanda]
                    -> Forza la modalità di analisi (senza domanda la fissa per le prossime)
  /auto             -> Torna alla scelta automatica della modalità
  /clear            -> Pulisce la memoria della conversazione
  /exit o /quit     -> Esci
================================================================================
"""

_CMD_RE = re.compile(r"^/[a-z]+$")


def handle_command(agent, text: str, history: list) -> bool:
    """Restituisce True se `text` era un comando locale (già gestito)."""
    parts = text.split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        raise SystemExit(0)
    if cmd == "/clear":
        history.clear()
        print("Memoria conversazione azzerata.")
        return True
    if cmd == "/map":
        print("\n" + agent.index.build_repo_map(4000))
        return True
    if cmd == "/onboarding":
        print("\n" + agent.onboarding())
        return True
    if cmd == "/outline":
        print("\n" + (agent.index.get_file_outline(arg) if arg else "Uso: /outline <file>"))
        return True
    if cmd == "/symbol":
        if not arg:
            print("Uso: /symbol <nome>")
            return True
        found = agent.index.find_symbol(arg)
        if not found:
            print(f"Nessun simbolo trovato per '{arg}'.")
        for c in found[:4]:
            print(f"\n[{c.path}:{c.start}-{c.end}] {c.kind} {c.qualname}\n{c.text[:1800]}")
        return True
    if cmd in MODE_COMMANDS:
        if arg:
            return False  # "/bug domanda": la gestisce agent.ask()
        if MODE_COMMANDS[cmd] is None:
            agent.unpin()
            print("Routing automatico riattivato.")
        else:
            agent.set_mode(MODE_COMMANDS[cmd], pin=True)
            print(f"Modalità fissata: {MODES[MODE_COMMANDS[cmd]].label}. /auto per tornare automatico.")
        return True
    if _CMD_RE.match(cmd):
        print(f"Comando sconosciuto: {cmd}. Disponibili: /outline /symbol /map /onboarding /teoria /pratica /bug /design /auto /clear /exit")
        return True
    return False


def main() -> None:
    if os.name == "nt":
        os.system("")  # abilita le sequenze ANSI nel terminale Windows

    providers_dict = load_providers_config()
    available_choices = list(providers_dict.keys()) + ["gemini", "groq"]

    ap = argparse.ArgumentParser(description="Repo Study Agent - CLI Interattiva per IDE")
    ap.add_argument("source", help="URL Git oppure percorso di una cartella locale")
    ap.add_argument(
        "--provider",
        default="gemini_3.8" if "gemini_3.8" in providers_dict else "gemini",
        choices=available_choices,
        help="Provider o alias definito in providers.json (es. gemini_3.8, groq_120b, cloudfare_glm-5.3)",
    )
    ap.add_argument(
        "--fallback",
        default="",
        help="Provider di riserva separati da virgola (alias di providers.json), usati se il principale va in rate limit. Es. groq_120b,gemini_3.6",
    )
    ap.add_argument("--model", default=None, help="ID del modello specifico (opzionale)")
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence_transformers", "gemini_api", "hashing"],
        help="Strategia di embedding",
    )
    ap.add_argument("--rebuild", action="store_true", help="Forza la ricostruzione totale dell'indice")
    ap.add_argument("--allow-exec", action="store_true", help="Abilita run_command: esegue comandi Python su una COPIA temporanea della repo")
    ap.add_argument("--allow-web", action="store_true", help="Abilita web_lookup: ricerca web e lettura di pagine (il contenuto è trattato come dato)")
    ap.add_argument("--no-cache", action="store_true", help="Disabilita la cache delle risposte (utile per misurare)")
    ap.add_argument("--no-fast-path", action="store_true", help="Disabilita il fast-path RAG a step singolo")
    ap.add_argument("--obs-budget", type=int, default=None, help="Caratteri di output dei tool tenuti in memoria prima di ridurre i più vecchi (0 = mai)")
    args = ap.parse_args()
    fallbacks = [a.strip() for a in args.fallback.split(",") if a.strip()]
    unknown = [a for a in fallbacks if a not in available_choices]
    if unknown:
        ap.error(f"--fallback: provider sconosciuti: {', '.join(unknown)}")

    cfg = Config(
        embedder_type=args.embedder, allow_exec=args.allow_exec, allow_web=args.allow_web,
        use_answer_cache=not args.no_cache,
        fast_path_enabled=not args.no_fast_path,
    )
    if args.obs_budget is not None:
        cfg.obs_budget_chars = args.obs_budget
    agent = create_agent(
        args.source,
        provider=args.provider,
        model_id=args.model,
        rebuild=args.rebuild,
        cfg=cfg,
        fallbacks=fallbacks or None,
    )

    print(HELP_BANNER)
    print(f"Repo attiva: {agent.index.repo_root.name} ({agent.index.stats()})\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            print()
            user_in = input("Domanda (o /comando) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit"):
            break
        if handle_command(agent, user_in, history):
            continue

        answer, trace = agent.ask(user_in, history)
        history.append((user_in, answer))
        print("\n" + answer)
        if trace:
            print("\nStrumenti usati: " + " → ".join(trace))
        if agent.last.cache_hit:
            print(agent.last.line())
        if agent.last.bad_citations:
            print("⚠️ Citazioni da ricontrollare: " + "; ".join(agent.last.bad_citations[:5]))


if __name__ == "__main__":
    main()