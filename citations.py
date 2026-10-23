"""Verifica e visualizzazione delle citazioni `percorso/file.ext:riga_inizio-riga_fine` (0 token LLM)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection

from ingest import CONFIG_EXTS, DOC_EXTS, EXT_TO_LANG, language_of

_CITE_RE = re.compile(
    r"(?<![\w./@+-])([\w@+-][\w./@+-]*\.[A-Za-z0-9]{1,6}):(\d+)(?:\s*[-–]\s*(\d+))?"
)
_KNOWN_EXTS = set(EXT_TO_LANG) | DOC_EXTS | CONFIG_EXTS


@dataclass
class CitationReport:
    valid: list[tuple[str, int, int]] = field(default_factory=list)   # (percorso nell'indice, inizio, fine)
    problems: list[str] = field(default_factory=list)


def _resolve(raw: str, files: Collection[str]) -> str | None:
    """Percorso esatto nell'indice, oppure unico file che finisce con `raw`; None se assente o ambiguo."""
    if raw in files:
        return raw
    matches = [f for f in files if f.endswith("/" + raw)]
    return matches[0] if len(matches) == 1 else None


def check_citations(
    answer: str, files: Collection[str], file_text: Callable[[str], str]
) -> CitationReport:
    """Controlla che ogni citazione punti a un file indicizzato e a righe che esistono davvero.

    Citazioni a file con estensione non riconosciuta (URL, numeri di versione) sono ignorate;
    quelle a file non presenti nell'indice sono segnalate (possono essere librerie esterne).
    """
    report = CitationReport()
    seen: set[tuple[str, int, int]] = set()
    for m in _CITE_RE.finditer(answer):
        raw, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        label = f"{raw}:{a}" + (f"-{b}" if m.group(3) else "")
        rel = _resolve(raw, files)
        if rel is None:
            if Path(raw).suffix.lower() in _KNOWN_EXTS and not any(f.endswith("/" + raw) for f in files):
                msg = f"{label} (file non presente nell'indice)"
                if msg not in report.problems:
                    report.problems.append(msg)
            continue
        if (rel, a, b) in seen:
            continue
        seen.add((rel, a, b))
        try:
            n = len(file_text(rel).split("\n"))
        except (ValueError, OSError):
            report.problems.append(f"{label} (file non leggibile)")
            continue
        if a < 1 or b < a or b > n:
            report.problems.append(f"{label} (righe fuori dal file: ne ha {n})")
        else:
            report.valid.append((rel, a, b))
    return report


def render_panel(
    report: CitationReport,
    file_text: Callable[[str], str],
    max_items: int = 5,
    max_lines: int = 30,
    max_chars: int = 6000,
) -> str:
    """Blocchi <details> con il codice citato (markdown per Gradio); stringa vuota se non c'è nulla."""
    blocks: list[str] = []
    used = 0
    for rel, a, b in report.valid[:max_items]:
        lines = file_text(rel).split("\n")
        end = min(b, a + max_lines - 1)
        code = "\n".join(lines[a - 1 : end])
        if used + len(code) > max_chars:
            break
        used += len(code)
        fence = "```"
        while fence in code:
            fence += "`"
        more = f" (mostrate {end - a + 1} righe su {b - a + 1})" if end < b else ""
        blocks.append(
            f"<details><summary>📄 {rel}:{a}-{b}{more}</summary>\n\n"
            f"{fence}{language_of(rel) or ''}\n{code}\n{fence}\n\n</details>"
        )
    if not blocks:
        return ""
    extra = len(report.valid) - len(blocks)
    tail = f"\n\n<sub>… altre {extra} citazioni non mostrate</sub>" if extra > 0 else ""
    return "\n\n" + "\n\n".join(blocks) + tail
