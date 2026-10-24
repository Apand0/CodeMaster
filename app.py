"""Fase 6: Interfaccia chat web con Gradio (locale, Colab o Kaggle) thread-safe."""
from __future__ import annotations

import re
import threading
from typing import Any, Callable

from agent import RepoAgent

EXAMPLES: list[str] = [
    "Dammi una panoramica: a cosa serve questa repository e come è organizzata?",
    "Qual è il punto di ingresso principale e come scorre l'esecuzione?",
    "Spiegami passo passo la funzione più importante del modulo principale.",
    "/onboarding",
    "/outline indexer.py",
]

# tutto ciò che segue questi marcatori (pannello citazioni, riga di telemetria) non va rimandato al modello come cronologia
_META_RE = re.compile(r"\n\n<(?:sub|details)>")


def _strip_meta(txt: str) -> str:
    return _META_RE.split(txt, maxsplit=1)[0]


def _text(content: Any) -> str:
    """Estrae il testo pulito da stringhe, dizionari o liste di blocchi Gradio."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, list):
        return " ".join(
            _text(b) if isinstance(b, (dict, list)) else str(b) for b in content
        )
    return str(content or "")


def history_to_pairs(history: list[Any] | None) -> list[tuple[str, str]]:
    """Normalizza la cronologia di Gradio ('messages' o 'tuples') in coppie (domanda, risposta)."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            txt = _text(item.get("content"))
            if role == "user":
                pending = txt
            elif role == "assistant" and pending is not None:
                pairs.append((pending, _strip_meta(txt)))
                pending = None
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((_text(item[0]), _strip_meta(_text(item[1]))))
    return pairs


def _handle_slash_command(agent: RepoAgent, cmd: str) -> str | None:
    """Esegue i comandi rapidi di ispezione AST a zero consumo di token LLM."""
    low = cmd.lower()
    if low == "/onboarding":
        return agent.onboarding()
    if low == "/map":
        return f"```text\n{agent.index.build_repo_map(4500)}\n```"
    if low.startswith("/outline "):
        target = cmd.split(" ", 1)[1].strip()
        return f"```text\n{agent.index.get_file_outline(target)}\n```"
    if low.startswith("/symbol "):
        sym = cmd.split(" ", 1)[1].strip()
        found = agent.index.find_symbol(sym)
        if not found:
            return f"Nessun simbolo trovato nell'indice per `{sym}`."
        blocks = [
            f"**[{c.path}:{c.start}-{c.end}] {c.kind} `{c.qualname}`**\n```{c.language or ''}\n{c.text[:1800]}\n```"
            for c in found[:4]
        ]
        return "\n\n".join(blocks)
    return None


def make_respond(agent: RepoAgent) -> Callable[[Any, list[Any] | None], str]:
    """Crea la callback di risposta protetta da Lock per evitare race condition sulla memoria dell'agente."""
    lock = threading.Lock()

    def respond(message: Any, history: list[Any] | None) -> str:
        clean_msg = _text(message).strip()
        if not clean_msg:
            return "Inserisci una domanda o incolla uno snippet di codice."

        if clean_msg.startswith("/"):
            fast_reply = _handle_slash_command(agent, clean_msg)
            if fast_reply is not None:
                return fast_reply + "\n\n<sub>⚡ Comando diretto AST (0 token API usati)</sub>"

        with lock:
            answer, trace = agent.ask(clean_msg, history_to_pairs(history))
            panel = agent.citation_panel(answer)
            stats = agent.last

        notes = []
        if trace:
            notes.append("🔧 Strumenti usati: " + " → ".join(trace))
        notes.append("📊 " + stats.line())
        if stats.bad_citations:
            notes.append("⚠️ Citazioni da ricontrollare: " + "; ".join(stats.bad_citations[:5]))
        return answer + panel + "\n\n<sub>" + "<br>".join(notes) + "</sub>"

    return respond


def build_ui(agent: RepoAgent) -> Any:
    """Costruisce il componente gr.ChatInterface compatibile con Gradio 4, 5 e 6."""
    import gradio as gr

    kwargs: dict[str, Any] = dict(
        fn=make_respond(agent),
        title=f"📚 Studia la repo: {agent.index.repo_root.name}",
        description=(
            f"Fai domande sul codice, incolla uno snippet o usa `/onboarding`, `/outline <file>`, `/symbol <nome>`, `/map`. "
            f"Indice: {agent.index.stats()}"
        ),
        examples=EXAMPLES,
    )
    try:
        return gr.ChatInterface(type="messages", **kwargs)  # Gradio 4/5
    except TypeError:
        return gr.ChatInterface(**kwargs)

def launch_chat(
    agent: RepoAgent,
    share: bool | None = None,
    server_port: int = 7860,
) -> Any:
    """ Costruisce e avvia l'interfaccia web Gradio in locale, Colab o Kaggle."""
    from config import detect_environment

    env = detect_environment()
    if share is None:
        share = env in ("colab", "kaggle")
    if share and getattr(agent, "backend", "api") == "cli":
        raise ValueError("La condivisione Gradio non è consentita con provider CLI autenticati localmente.")

    ui = build_ui(agent)
    return ui.launch(
        share=share,
         inline=(env in ("colab", "kaggle")),
        server_port=server_port,
    )
