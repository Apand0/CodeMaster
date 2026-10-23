"""Cache delle risposte (0 token su domande ripetute), legata agli SHA-256 dei file citati."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

MAX_ENTRIES = 200


def make_key(question: str, mode: str, model_id: str, prompt_sig: str, prev_questions: list[str]) -> str:
    """Chiave stabile: domanda normalizzata + modalità + modello + versione dei prompt + domande precedenti."""
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    raw = "\x1f".join([mode, model_id, prompt_sig, norm(question), *map(norm, prev_questions)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class AnswerCache:
    """Voce = risposta + {file citato: sha256}. Un hit richiede che ogni file abbia ancora lo stesso hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._data: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._data = {}

    def get(self, key: str, sha_of: Callable[[str], str]) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if all(sha_of(rel) == sha for rel, sha in entry["deps"].items()):
            return entry["answer"]
        del self._data[key]          # un file citato è cambiato: la risposta è da rifare
        self._save()
        return None

    def put(self, key: str, question: str, answer: str, deps: dict[str, str]) -> None:
        self._data[key] = {"q": question[:200], "answer": answer, "deps": deps, "ts": time.time()}
        for old in sorted(self._data, key=lambda k: self._data[k]["ts"])[: max(0, len(self._data) - MAX_ENTRIES)]:
            del self._data[old]
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # la cache è un'ottimizzazione: un disco non scrivibile non deve rompere la risposta
