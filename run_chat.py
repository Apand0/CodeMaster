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
    ap.add_argument("--model", default=None, help="model_id specifico")
    ap.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "sentence_transformers", "gemini_api", "hashing"],
        help="Strategia di embedding",
    )
    ap.add_argument("--rebuild", action="store_true", help="Forza la ricostruzione completa dell'indice")
    ap.add_argument("--share", action="store_true", help="Genera un link pubblico temporaneo Gradio")
    ap.add_argument("--port", type=int, default=7860, help="Porta locale per il server Gradio")
    args = ap.parse_args()

    cfg = Config(embedder_type=args.embedder)
    agent = create_agent(
        args.source,
        provider=args.provider,
        model_id=args.model,
        rebuild=args.rebuild,
        cfg=cfg,
    )
    launch_chat(agent, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()