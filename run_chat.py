"""Avvio dell'interfaccia web Gradio da terminale."""
from __future__ import annotations

import argparse

from agent import create_agent
from app import launch_chat
from config import Config, load_providers_config


def main() -> None:
    providers_dict = load_providers_config()
    available_choices = list(providers_dict.keys()) + ["gemini", "groq"]

    ap = argparse.ArgumentParser(description="Avvia la chat web Gradio per studiare una repository")
    ap.add_argument("source", help="URL Git oppure percorso di una cartella locale")
    ap.add_argument(
        "--provider",
        default="gemini_3.8" if "gemini_3.8" in providers_dict else "gemini",
        choices=available_choices,
        help="Provider o alias da utilizzare (es. da providers.json)",
    )
    ap.add_argument(
        "--fallback",
        default="",
        help="Provider di riserva separati da virgola (alias di providers.json), usati se il principale va in rate limit. Es. groq_120b,gemini_3.6",
    )
    ap.add_argument("--model", default=None, help="model_id specifico")
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence_transformers", "gemini_api", "hashing"],
        help="Strategia di embedding",
    )
    ap.add_argument("--rebuild", action="store_true", help="Forza la ricostruzione completa dell'indice")
    ap.add_argument("--allow-exec", action="store_true", help="Abilita run_command: esegue comandi Python su una COPIA temporanea della repo")
    ap.add_argument("--allow-web", action="store_true", help="Abilita web_lookup: ricerca web e lettura di pagine (il contenuto è trattato come dato)")
    ap.add_argument("--no-cache", action="store_true", help="Disabilita la cache delle risposte (utile per misurare)")
    ap.add_argument("--no-fast-path", action="store_true", help="Disabilita il fast-path RAG a step singolo")
    ap.add_argument("--obs-budget", type=int, default=None, help="Caratteri di output dei tool tenuti in memoria prima di ridurre i più vecchi (0 = mai)")
    ap.add_argument("--share", action="store_true", help="Genera un link pubblico temporaneo Gradio")
    ap.add_argument("--port", type=int, default=7860, help="Porta locale per il server Gradio")
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
    launch_chat(agent, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()