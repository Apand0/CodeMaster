"""CLI interattiva per usare l'agente direttamente nel terminale del tuo IDE (VS Code, PyCharm, ecc.)."""
from __future__ import annotations

import argparse
import sys

from agent import create_agent
from config import Config, load_providers_config

HELP_BANNER = """
================================================================================
🚀 REPO STUDY AGENT — Modalità Terminale / IDE (Con supporto providers.json)
Comandi rapidi disponibili in chat:
  /outline <file>   -> Mostra subito la struttura AST di un file (0 token LLM)
  /symbol <nome>    -> Trova subito la definizione di una funzione/classe (0 token LLM)
  /map              -> Mostra la mappa generale della repository
  /clear            -> Pulisce la memoria della conversazione
  /exit o /quit     -> Esci
================================================================================
"""


def main() -> None:
    providers_dict = load_providers_config()
    available_choices = list(providers_dict.keys()) + ["gemini", "groq"]

    ap = argparse.ArgumentParser(description="Repo Study Agent — CLI Interattiva per IDE")
    ap.add_argument("source", help="URL Git oppure percorso di una cartella locale")
    ap.add_argument(
        "--provider",
        default="gemini_3.8" if "gemini_3.8" in providers_dict else "gemini",
        choices=available_choices,
        help="Provider o alias definito in providers.json (es. gemini_3.8, groq_120b, cloudfare_glm-5.3)",
    )
    ap.add_argument("--model", default=None, help="ID del modello specifico (opzionale)")
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence_transformers", "gemini_api", "hashing"],
        help="Strategia di embedding",
    )
    ap.add_argument("--rebuild", action="store_true", help="Forza la ricostruzione totale dell'indice")
    args = ap.parse_args()

    cfg = Config(embedder_type=args.embedder)
    agent = create_agent(
        args.source,
        provider=args.provider,
        model_id=args.model,
        rebuild=args.rebuild,
        cfg=cfg,
    )

    print(HELP_BANNER)
    print(f"📂 Repo attiva: {agent.index.repo_root.name} ({agent.index.stats()})\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_in = input("\n🟢 Domanda (o /comando) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            sys.exit(0)

        if not user_in:
            continue
        if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if user_in.lower() == "/clear":
            history.clear()
            print("Memoria conversazione azzerata.")
            continue
        if user_in.lower() == "/map":
            print("\n" + agent.index.build_repo_map(4000))
            continue
        if user_in.lower().startswith("/outline "):
            target = user_in.split(" ", 1)[1].strip()
            print("\n" + agent.index.get_file_outline(target))
            continue
        if user_in.lower().startswith("/symbol "):
            sym = user_in.split(" ", 1)[1].strip()
            found = agent.index.find_symbol(sym)
            if not found:
                print(f"Nessun simbolo trovato per '{sym}'.")
            for c in found[:4]:
                print(f"\n[{c.path}:{c.start}-{c.end}] {c.kind} {c.qualname}\n{c.text[:1800]}")
            continue

        answer, trace = agent.ask(user_in, history)
        history.append((user_in, answer))
        print("\n" + answer)
        if trace:
            print("\n🔧 Strumenti usati: " + " → ".join(trace))


if __name__ == "__main__":
    main()