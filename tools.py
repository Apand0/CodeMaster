"""Fase 4: Strumenti dell'agente (inclusi get_file_outline e locate_code_snippet)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from smolagents import Tool

from chunker import Chunk, render_notebook
from indexer import RepoIndex
from ingest import read_text


class RepoContext:
    """Stato condiviso e sandboxed per i tool di esplorazione della repository."""

    def __init__(self, index: RepoIndex) -> None:
        self.index = index
        self.cfg = index.cfg
        self.root = index.repo_root.resolve()

    def safe_path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"Percorso fuori dalla repository: {rel}")
        return p

    @staticmethod
    def normalize_rel_prefix(path: str | None) -> str:
        if not path or path.strip() in (".", "./", "/"):
            return ""
        cleaned = path.strip().replace("\\", "/")
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        return cleaned.lstrip("/")

    def clip(self, text: str) -> str:
        n = self.cfg.max_tool_output_chars
        return text if len(text) <= n else text[:n] + f"\n... [output troncato a {n} caratteri]"

    def file_text(self, rel: str) -> str:
        p = self.safe_path(rel)
        raw = read_text(p)
        return render_notebook(raw) if p.suffix == ".ipynb" else raw


def _fmt_chunk(c: Chunk, with_text: bool = True, max_text: int = 1400) -> str:
    head = f"[{c.path}:{c.start}-{c.end}] {c.kind} {c.qualname}"
    if c.signature:
        head += f"\n  firma: {c.signature}"
    if c.docstring:
        head += f"\n  doc: {c.docstring[:300]}"
    if not with_text:
        return head
    body = c.text if len(c.text) <= max_text else c.text[:max_text] + "\n  ... [usa read_file per il resto]"
    return head + "\n" + body


class _Base(Tool):
    skip_forward_signature_validation = False

    def __init__(self, ctx: RepoContext) -> None:
        super().__init__()
        self.ctx = ctx


class ListDirTool(_Base):
    name = "list_dir"
    description = "Elenca file e sottocartelle di una directory della repository."
    inputs = {
        "path": {
            "type": "string",
            "description": "Cartella relativa alla radice della repo ('.' per la radice).",
            "nullable": True,
        }
    }
    output_type = "string"

    def forward(self, path: str = ".") -> str:
        try:
            p = self.ctx.safe_path(path or ".")
        except ValueError as e:
            return str(e)
        if not p.is_dir():
            return f"'{path}' non è una cartella."
        skip = set(self.ctx.cfg.skip_dirs)
        entries: list[str] = []
        for child in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if child.name in skip:
                continue
            if child.is_dir():
                rel_dir = child.relative_to(self.ctx.root).as_posix() + "/"
                n = sum(1 for f in self.ctx.index.files if f.startswith(rel_dir))
                entries.append(f"{child.name}/  ({n} file indicizzati)")
            else:
                entries.append(child.name)
        return self.ctx.clip("\n".join(entries) or "(vuota)")


class FileOutlineTool(_Base):
    name = "get_file_outline"
    description = (
        "Mostra la radiografia completa di un file: tutti gli import, classi, funzioni/metodi "
        "con le loro firme, quali funzioni chiamano e i numeri di riga esatti. "
        "Usalo SEMPRE prima di read_file per orientarti in file lunghi o privi di documentazione."
    )
    inputs = {"path": {"type": "string", "description": "Percorso relativo del file."}}
    output_type = "string"

    def forward(self, path: str) -> str:
        return self.ctx.clip(self.ctx.index.get_file_outline(path))


class ReadFileTool(_Base):
    name = "read_file"
    description = (
        "Legge un intervallo di righe di un file con numeri di riga (max 250 righe per chiamata). "
        "Usa get_file_outline prima per sapere quali righe leggere."
    )
    inputs = {
        "path": {"type": "string", "description": "Percorso relativo del file."},
        "start_line": {"type": "integer", "description": "Prima riga (1-based). Default 1.", "nullable": True},
        "end_line": {"type": "integer", "description": "Ultima riga inclusa. Default start+199.", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        try:
            text = self.ctx.file_text(path)
        except (ValueError, OSError) as e:
            return f"Errore: {e}"
        lines = text.split("\n")
        s = max(1, int(start_line or 1))
        e = int(end_line) if end_line else s + 199
        e = min(e, s + 249, len(lines))
        if s > len(lines):
            return f"Il file ha solo {len(lines)} righe."
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(s, e + 1))
        return self.ctx.clip(f"{path} (righe {s}-{e} di {len(lines)})\n{body}")


class LocateSnippetTool(_Base):
    name = "locate_code_snippet"
    description = (
        "Trova il file esatto, i numeri di riga e il simbolo AST contenitore per qualsiasi "
        "blocco o riga di codice incollata dall'utente."
    )
    inputs = {"snippet": {"type": "string", "description": "Frammento di codice incollato dall'utente."}}
    output_type = "string"

    def forward(self, snippet: str) -> str:
        raw_lines = [ln.strip() for ln in snippet.strip().splitlines() if ln.strip()]
        if not raw_lines:
            return "Snippet vuoto."
        # Sceglie la riga più significativa (più lunga e non banale) come ancora
        anchor = max(raw_lines, key=len)
        matches: list[str] = []
        for rel in self.ctx.index.files:
            try:
                content = self.ctx.file_text(rel)
            except OSError:
                continue
            if anchor not in content:
                continue
            file_lines = content.splitlines()
            for idx, fline in enumerate(file_lines, 1):
                if anchor in fline:
                    s_line = max(1, idx - 5)
                    e_line = min(len(file_lines), idx + len(raw_lines) + 10)
                    enclosing = [
                        c.qualname
                        for c in self.ctx.index.by_file.get(rel, [])
                        if c.start <= idx <= c.end and c.kind != "module"
                    ]
                    sym_info = f" (dentro {', '.join(enclosing)})" if enclosing else ""
                    excerpt = "\n".join(f"{ln_no:>5}| {file_lines[ln_no - 1]}" for ln_no in range(s_line, e_line + 1))
                    matches.append(f"Trovato in `{rel}:{idx}`{sym_info}:\n{excerpt}")
                    if len(matches) >= 5:
                        break
            if len(matches) >= 5:
                break
        if matches:
            return self.ctx.clip("\n\n".join(matches))
        # Fallback su ricerca ibrida se lo snippet è stato modificato o riformattato
        res = self.ctx.index.search(snippet, k=4)
        if not res:
            return "Impossibile trovare lo snippet nella repository."
        return self.ctx.clip(
            "Match esatto non trovato; ecco i blocchi più simili:\n\n"
            + "\n\n".join(_fmt_chunk(c, True, 1200) for c, _ in res)
        )


class SearchCodeTool(_Base):
    name = "search_code"
    description = (
        "Ricerca ibrida (semantica + BM25) nel codice e nella documentazione. "
        "Supporta path_prefix per filtrare la ricerca dentro una cartella o un singolo file."
    )
    inputs = {
        "query": {"type": "string", "description": "Domanda concettuale o termini tecnici."},
        "top_k": {"type": "integer", "description": "Numero di risultati (default 6, max 12).", "nullable": True},
        "path_prefix": {"type": "string", "description": "Limita la ricerca a una cartella o file.", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, top_k: int = 6, path_prefix: str | None = None) -> str:
        k = max(1, min(int(top_k or 6), 12))
        res = self.ctx.index.search(query, k=k, path_prefix=path_prefix or None)
        if not res:
            return "Nessun risultato."
        per = max(400, self.ctx.cfg.max_tool_output_chars // k - 150)
        return self.ctx.clip("\n\n".join(_fmt_chunk(c, True, per) for c, _ in res))


class GrepTool(_Base):
    name = "grep"
    description = "Cerca un'espressione regolare o stringa (case-insensitive) nei file della repository."
    inputs = {
        "pattern": {"type": "string", "description": "Regex o testo esatto da cercare."},
        "path": {"type": "string", "description": "Cartella o file in cui cercare (default: tutta la repo).", "nullable": True},
        "max_results": {"type": "integer", "description": "Massimo risultati (default 40).", "nullable": True},
    }
    output_type = "string"

    def forward(self, pattern: str, path: str = ".", max_results: int = 40) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        prefix = self.ctx.normalize_rel_prefix(path)
        out: list[str] = []
        limit = max(1, int(max_results or 40))
        for rel in self.ctx.index.files:
            if prefix and not rel.startswith(prefix):
                continue
            try:
                lines = self.ctx.file_text(rel).split("\n")
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    out.append(f"{rel}:{i}: {line.strip()[:160]}")
                    if len(out) >= limit:
                        return self.ctx.clip("\n".join(out) + "\n... [limite raggiunto]")
        return self.ctx.clip("\n".join(out)) if out else "Nessuna corrispondenza."


class FindDefinitionTool(_Base):
    name = "find_definition"
    description = "Trova dove è definita una funzione, classe o metodo (es. 'forward' o 'GPT.generate') e mostra il codice."
    inputs = {"symbol": {"type": "string", "description": "Nome del simbolo o 'Classe.metodo'."}}
    output_type = "string"

    def forward(self, symbol: str) -> str:
        found = self.ctx.index.find_symbol(symbol)
        if not found:
            return f"Nessuna definizione trovata per '{symbol}'. Prova search_code o grep."
        shown = found[:4]
        txt = "\n\n".join(_fmt_chunk(c, True, 2500) for c in shown)
        if len(found) > len(shown):
            txt += "\n\nAltre parti/definizioni: " + ", ".join(
                f"{c.path}:{c.start}-{c.end} {c.qualname}" for c in found[4:12]
            )
        return self.ctx.clip(txt)


class FindReferencesTool(_Base):
    name = "find_references"
    description = "Elenca le righe della repository in cui un simbolo viene utilizzato (match su parola intera)."
    inputs = {
        "symbol": {"type": "string", "description": "Nome esatto del simbolo."},
        "max_results": {"type": "integer", "description": "Massimo risultati (default 35).", "nullable": True},
    }
    output_type = "string"

    def forward(self, symbol: str, max_results: int = 35) -> str:
        rx = re.compile(rf"\b{re.escape(symbol.split('.')[-1].strip())}\b")
        out: list[str] = []
        limit = max(1, int(max_results or 35))
        for rel in self.ctx.index.files:
            try:
                lines = self.ctx.file_text(rel).split("\n")
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    out.append(f"{rel}:{i}: {line.strip()[:160]}")
                    if len(out) >= limit:
                        return self.ctx.clip("\n".join(out) + "\n... [limite raggiunto]")
        return self.ctx.clip("\n".join(out)) if out else f"Nessun riferimento a '{symbol}'."


class CallGraphTool(_Base):
    name = "get_call_graph"
    description = (
        "Mostra chi chiama una funzione (callers) e quali funzioni essa chiama (callees), "
        "dando priorità ai chiamanti nello stesso file o che importano il modulo."
    )
    inputs = {"function_name": {"type": "string", "description": "Nome della funzione o del metodo."}}
    output_type = "string"

    def forward(self, function_name: str) -> str:
        idx = self.ctx.index
        defs = [
            c for c in idx.find_symbol(function_name)
            if c.kind in ("function", "method") and ("[parte " not in c.qualname or "[parte 1/" in c.qualname)
        ]
        def_files = {d.path for d in defs}
        def_stems = {Path(d.path).stem for d in defs}

        short = function_name.split(".")[-1]
        raw_callers = idx.callers.get(function_name, []) + [
            c for c in idx.callers.get(short, []) if c not in idx.callers.get(function_name, [])
        ]

        # Ordina i chiamanti dando priorità a quelli nello stesso file o che importano il modulo
        def caller_priority(c: Chunk) -> tuple[int, str, int]:
            same_file = 0 if c.path in def_files else 1
            imports_target = 0 if any(stem in " ".join(c.imports) for stem in def_stems) else 1
            return (same_file, imports_target, c.path, c.start)

        callers = sorted(raw_callers, key=caller_priority)

        lines: list[str] = []
        for d in defs[:4]:
            lines.append(
                f"Definizione {d.qualname} [{d.path}:{d.start}-{d.end}] chiama: "
                f"{', '.join(d.calls) or '(nessuna chiamata rilevata)'}"
            )
        lines.append(f"\nChiamata da ({len(callers)}):")
        for c in callers[:25]:
            tag = " [stesso file]" if c.path in def_files else ""
            lines.append(f"  <- {c.qualname} [{c.path}:{c.start}-{c.end}]{tag}")
        if not defs and not callers:
            return f"Nessun dato per '{function_name}'."
        return self.ctx.clip("\n".join(lines))


class GitLogTool(_Base):
    name = "git_log"
    description = "Ultimi commit Git (autore, data, messaggio) che hanno toccato un file o la repository."
    inputs = {
        "path": {"type": "string", "description": "File o cartella (default: '.').", "nullable": True},
        "n": {"type": "integer", "description": "Numero di commit (default 10, max 30).", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str = ".", n: int = 10) -> str:
        if not (self.ctx.root / ".git").exists():
            return "La cartella non è una repository git."
        n = max(1, min(int(n or 10), 30))
        try:
            p = self.ctx.safe_path(path or ".")
            rel = p.relative_to(self.ctx.root).as_posix() if p != self.ctx.root else "."
            out = subprocess.run(
                [
                    "git", "-C", str(self.ctx.root), "log", f"-{n}",
                    "--date=short", "--pretty=format:%h %ad %an: %s", "--", rel,
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).stdout
        except Exception as e:
            return f"Errore git: {e}"
        return self.ctx.clip(out.strip() or "Nessun commit trovato.")


def build_tools(index: RepoIndex) -> list[Tool]:
    ctx = RepoContext(index)
    return [
        cls(ctx)
        for cls in (
            ListDirTool,
            FileOutlineTool,
            ReadFileTool,
            LocateSnippetTool,
            SearchCodeTool,
            GrepTool,
            FindDefinitionTool,
            FindReferencesTool,
            CallGraphTool,
            GitLogTool,
        )
    ]