"""Fase 2: Parsing AST con tree-sitter (funzioni, classi, import, call-graph qualificato)."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import Config, logger


@dataclass
class Chunk:
    """Unità atomica di codice o documentazione estratta da un file."""

    id: str
    path: str
    kind: str                 # function | method | class | struct | module | doc | config | text
    name: str                 # es. "forward"
    qualname: str             # es. "GPT.forward"
    start: int                # riga iniziale (1-based)
    end: int                  # riga finale (inclusa)
    language: str | None = None
    signature: str = ""
    docstring: str = ""
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        d.setdefault("calls", [])
        d.setdefault("imports", [])
        return cls(**d)


_JS_FUNCS = {"function_declaration", "generator_function_declaration", "method_definition"}

LANG_SPECS: dict[str, dict[str, Any]] = {
    "python": {
        "containers": {"class_definition": "class"},
        "functions": {"function_definition"},
        "leaves": {},
        "imports": {"import_statement", "import_from_statement"},
    },
    "javascript": {
        "containers": {"class_declaration": "class"},
        "functions": _JS_FUNCS,
        "leaves": {},
        "imports": {"import_statement"},
        "arrow": True,
    },
    "typescript": {
        "containers": {
            "class_declaration": "class",
            "abstract_class_declaration": "class",
            "interface_declaration": "interface",
        },
        "functions": _JS_FUNCS,
        "leaves": {},
        "imports": {"import_statement"},
        "arrow": True,
    },
    "tsx": {
        "containers": {"class_declaration": "class", "interface_declaration": "interface"},
        "functions": _JS_FUNCS,
        "leaves": {},
        "imports": {"import_statement"},
        "arrow": True,
    },
    "java": {
        "containers": {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "class",
        },
        "functions": {"method_declaration", "constructor_declaration"},
        "leaves": {},
        "imports": {"import_declaration"},
    },
    "go": {
        "containers": {},
        "functions": {"function_declaration", "method_declaration"},
        "leaves": {"type_declaration": "struct"},
        "imports": {"import_declaration"},
    },
    "rust": {
        "containers": {"impl_item": "impl", "trait_item": "trait"},
        "functions": {"function_item"},
        "leaves": {"struct_item": "struct", "enum_item": "enum"},
        "imports": {"use_declaration"},
    },
    "c": {
        "containers": {},
        "functions": {"function_definition"},
        "leaves": {},
        "imports": {"preproc_include"},
    },
    "cpp": {
        "containers": {"class_specifier": "class", "struct_specifier": "struct"},
        "functions": {"function_definition"},
        "leaves": {},
        "imports": {"preproc_include"},
    },
    "csharp": {
        "containers": {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "struct_declaration": "struct",
        },
        "functions": {"method_declaration", "constructor_declaration"},
        "leaves": {},
        "imports": {"using_directive"},
    },
    "ruby": {
        "containers": {"class": "class", "module": "class"},
        "functions": {"method", "singleton_method"},
        "leaves": {},
        "imports": set(),
    },
    "php": {
        "containers": {"class_declaration": "class"},
        "functions": {"function_definition", "method_declaration"},
        "leaves": {},
        "imports": {"namespace_use_declaration"},
    },
    "kotlin": {
        "containers": {"class_declaration": "class", "object_declaration": "class"},
        "functions": {"function_declaration"},
        "leaves": {},
        "imports": {"import_header"},
    },
    "swift": {
        "containers": {"class_declaration": "class", "protocol_declaration": "interface"},
        "functions": {"function_declaration"},
        "leaves": {},
        "imports": {"import_declaration"},
    },
    "scala": {
        "containers": {
            "class_definition": "class",
            "object_definition": "class",
            "trait_definition": "interface",
        },
        "functions": {"function_definition"},
        "leaves": {},
        "imports": {"import_declaration"},
    },
    "bash": {"containers": {}, "functions": {"function_definition"}, "leaves": {}, "imports": set()},
    "lua": {"containers": {}, "functions": {"function_declaration"}, "leaves": {}, "imports": set()},
}

CALL_TYPES = {
    "call", "call_expression", "method_invocation", "invocation_expression",
    "function_call_expression", "member_call_expression",
}
ARROW_VALUE_TYPES = {"arrow_function", "function_expression", "function", "generator_function"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEADING_PREFIX_RE = re.compile(r"^(?:#+\s*|# %%\s*\[markdown\]\s*|# %%\s*)", re.IGNORECASE)

_PARSERS: dict[str, Any] = {}


def _get_parser(lang: str) -> Any:
    if lang in _PARSERS:
        return _PARSERS[lang]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
    except Exception as exc:
        logger.debug("Parser non disponibile per %s: %s", lang, exc)
    _PARSERS[lang] = parser
    return parser


def windows(
    lines: list[str], size: int, overlap: int, offset: int = 0
) -> list[tuple[int, int, str]]:
    """Finestre di righe sovrapposte con indici 1-based."""
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(lines)
    step = max(1, size - overlap)
    while i < n:
        j = min(n, i + size)
        text = "\n".join(lines[i:j])
        if text.strip():
            out.append((offset + i + 1, offset + j, text))
        if j >= n:
            break
        i += step
    return out


def split_long(chunk: Chunk, cfg: Config) -> list[Chunk]:
    """Spezza chunk lunghi in parti mantenendo i numeri di riga reali."""
    if len(chunk.text) <= cfg.max_chunk_chars:
        return [chunk]
    lines = chunk.text.split("\n")
    parts = windows(lines, cfg.window_lines, cfg.window_overlap, offset=chunk.start - 1)
    out: list[Chunk] = []
    for k, (s, e, t) in enumerate(parts, 1):
        data = chunk.to_dict()
        data.update({
            "start": s,
            "end": e,
            "text": t,
            "qualname": f"{chunk.qualname} [parte {k}/{len(parts)}]",
        })
        c = Chunk.from_dict(data)
        if k > 1:
            c.docstring = ""
        out.append(c)
    return out


def render_notebook(raw: str) -> str:
    """Converte un file .ipynb in testo lineare con marcatori di cella."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    out: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src_text = "".join(src) if isinstance(src, list) else str(src)
        marker = "# %% [markdown]" if cell.get("cell_type") == "markdown" else "# %%"
        out.extend([marker, src_text.rstrip(), ""])
    return "\n".join(out)


class _CodeChunker:
    """Estrattore AST per codice sorgente tramite Tree-sitter."""

    def __init__(self, path: str, source: str, lang: str, cfg: Config) -> None:
        self.path = path
        self.lang = lang
        self.cfg = cfg
        self.spec = LANG_SPECS[lang]
        self.src = source.encode("utf-8")
        self.lines = source.split("\n")
        self.chunks: list[Chunk] = []
        self.file_imports: list[str] = []

    def _slice(self, node: Any) -> str:
        return self.src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _lines_of(self, node: Any) -> tuple[int, int, str]:
        s, e = node.start_point[0], node.end_point[0]
        return s + 1, e + 1, "\n".join(self.lines[s : e + 1])

    def _name(self, node: Any) -> str:
        for fld in ("name", "type", "declarator"):
            n = node.child_by_field_name(fld)
            if n is not None:
                txt = self._slice(n).split("(")[0].split("<")[0].strip()
                if m := _IDENT.findall(txt):
                    return m[-1]
        if node.type == "type_declaration":
            for ch in node.named_children:
                if (n := ch.child_by_field_name("name")) is not None:
                    return self._slice(n)
        for ch in node.named_children:
            if ch.type in ("identifier", "type_identifier", "name", "constant", "simple_identifier"):
                return self._slice(ch)
        return "<anonimo>"

    def _signature(self, node: Any) -> str:
        body = node.child_by_field_name("body")
        if body is not None and body.start_byte > node.start_byte:
            sig = self.src[node.start_byte:body.start_byte].decode("utf-8", errors="replace")
        else:
            sig = self._slice(node).split("\n")[0]
        sig = re.sub(r"(?m)^\s*(#|//).*$", "", sig)
        sig = re.sub(r"\s+", " ", sig).strip()
        return sig[:260]

    def _docstring(self, node: Any, outer: Any) -> str:
        if self.lang == "python":
            body = node.child_by_field_name("body")
            if body is not None and body.named_children:
                first = body.named_children[0]
                lit = None
                if first.type in ("string", "concatenated_string"):
                    lit = first
                elif (
                    first.type == "expression_statement"
                    and first.named_children
                    and first.named_children[0].type in ("string", "concatenated_string")
                ):
                    lit = first.named_children[0]
                if lit is not None:
                    raw = self._slice(lit)
                    try:
                        val = ast.literal_eval(raw)
                        if isinstance(val, str):
                            return val.strip()[:800]
                    except Exception:
                        return raw.strip("\"' \n")[:800]

        ref = outer or node
        comments: list[str] = []
        curr = ref.prev_sibling
        expected_row = ref.start_point[0] - 1
        while (
            curr is not None
            and "comment" in curr.type
            and curr.end_point[0] >= expected_row - 1
            and (curr.prev_sibling is None or curr.prev_sibling.end_point[0] < curr.start_point[0])
        ):
            txt = self._slice(curr)
            txt = re.sub(r"^\s*(/\*+|\*/|//+|#+|\*)\s?", "", txt, flags=re.M).strip()
            if txt:
                comments.append(txt)
            expected_row = curr.start_point[0] - 1
            curr = curr.prev_sibling
        if comments:
            return "\n".join(reversed(comments))[:800]
        return ""

    def _calls(self, node: Any) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type in CALL_TYPES:
                target = n.child_by_field_name("function") or n.child_by_field_name("name")
                if target is None and n.named_children:
                    target = n.named_children[0]
                if target is not None:
                    ids = _IDENT.findall(self._slice(target))
                    if ids:
                        short = ids[-1]
                        if short not in seen:
                            seen.add(short)
                            names.append(short)
                        # Se è una chiamata qualificata (es. module.func), salva anche il qualificatore
                        if len(ids) >= 2 and ids[-2] not in ("self", "this", "cls", "super"):
                            qual_call = f"{ids[-2]}.{ids[-1]}"
                            if qual_call not in seen:
                                seen.add(qual_call)
                                names.append(qual_call)
            stack.extend(reversed(n.children))
        return names[:80]

    def _make(
        self, node: Any, outer: Any, kind: str, name: str, parents: list[str], with_calls: bool
    ) -> Chunk:
        rng = outer or node
        s, e, text = self._lines_of(rng)
        qual = ".".join(parents + [name]) if parents else name
        return Chunk(
            id="",
            path=self.path,
            kind=kind,
            name=name,
            qualname=qual,
            start=s,
            end=e,
            language=self.lang,
            signature=self._signature(node),
            docstring=self._docstring(node, outer),
            calls=self._calls(node) if with_calls else [],
            imports=list(self.file_imports),
            text=text,
        )

    def _walk(self, node: Any, parents: list[str], in_class: bool, outer: Any = None) -> None:
        t = node.type
        spec = self.spec

        if t in spec.get("imports", set()):
            imp_stmt = re.sub(r"\s+", " ", self._slice(node)).strip()
            if imp_stmt and len(self.file_imports) < 60:
                self.file_imports.append(imp_stmt[:180])
            return

        if t == "decorated_definition":
            if (inner := node.child_by_field_name("definition")) is not None:
                self._walk(inner, parents, in_class, outer=node)
            return

        if t in spec["functions"]:
            name = self._name(node)
            kind = "method" if in_class else "function"
            self.chunks.extend(split_long(self._make(node, outer, kind, name, parents, True), self.cfg))
            return

        if t in spec["containers"]:
            name = self._name(node)
            before = len(self.chunks)
            for ch in node.named_children:
                self._walk(ch, parents + [name], True)
            kids = self.chunks[before:]
            s, e, _ = self._lines_of(outer or node)
            first_child_start = min((c.start for c in kids), default=e + 1)
            head_end = min(e, first_child_start - 1, s + 39)
            head = "\n".join(self.lines[s - 1 : max(s, head_end)])
            sigs = "\n".join(
                f"    # {c.signature}" for c in kids if c.kind == "method" and "[parte " not in c.qualname
            )
            text = head + (f"\n    # --- metodi ---\n{sigs}" if sigs else "")
            ch = self._make(node, outer, spec["containers"][t], name, parents, False)
            ch.text = text if kids else "\n".join(self.lines[s - 1 : e])
            ch.end = max(s, head_end) if kids else e
            parts = split_long(ch, self.cfg)
            self.chunks.insert(before, parts[0])
            self.chunks.extend(parts[1:])
            return

        if t in spec["leaves"]:
            name = self._name(node)
            self.chunks.extend(
                split_long(self._make(node, outer, spec["leaves"][t], name, parents, False), self.cfg)
            )
            return

        if spec.get("arrow") and t in ("lexical_declaration", "variable_declaration") and not in_class:
            handled = False
            for d in node.named_children:
                if d.type == "variable_declarator":
                    val = d.child_by_field_name("value")
                    if val is not None and val.type in ARROW_VALUE_TYPES:
                        name = self._name(d)
                        ch = self._make(val, node, "function", name, parents, True)
                        ch.signature = re.sub(r"\s+", " ", self._slice(node).split("=>")[0])[:240]
                        self.chunks.extend(split_long(ch, self.cfg))
                        handled = True
            if handled:
                return

        for ch in node.named_children:
            self._walk(ch, parents, in_class)

    def run(self) -> list[Chunk]:
        parser = _get_parser(self.lang)
        if parser is None:
            return []
        tree = parser.parse(self.src)
        self._walk(tree.root_node, [], False)
        for c in self.chunks:
            c.imports = list(self.file_imports)
        self._add_module_chunks()
        return self.chunks

    def _add_module_chunks(self) -> None:
        covered: set[int] = set()
        for c in self.chunks:
            if c.kind in ("function", "method", "class", "struct", "interface", "enum", "impl", "trait"):
                covered.update(range(c.start, c.end + 1))

        segments: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for i, line in enumerate(self.lines, 1):
            if i in covered:
                if cur:
                    segments.append(cur)
                    cur = []
            else:
                cur.append((i, line))
        if cur:
            segments.append(cur)

        group: list[list[tuple[int, str]]] = []
        size = 0

        def flush() -> None:
            nonlocal group, size
            if not group:
                return
            text = "\n# ...\n".join("\n".join(l for _, l in seg) for seg in group)
            if text.strip():
                first, last = group[0][0][0], group[-1][-1][0]
                ch = Chunk(
                    id="",
                    path=self.path,
                    kind="module",
                    name="<modulo>",
                    qualname="<modulo>",
                    start=first,
                    end=last,
                    language=self.lang,
                    imports=list(self.file_imports),
                    text=text,
                )
                if len(text) > self.cfg.max_chunk_chars and len(group) == 1:
                    lines = [l for _, l in group[0]]
                    for s, e, t in windows(lines, self.cfg.window_lines, self.cfg.window_overlap, offset=first - 1):
                        data = ch.to_dict()
                        data.update({"start": s, "end": e, "text": t})
                        self.chunks.append(Chunk.from_dict(data))
                else:
                    self.chunks.append(ch)
            group, size = [], 0

        for seg in segments:
            seg_text = "\n".join(l for _, l in seg)
            if not seg_text.strip():
                continue
            if size + len(seg_text) > self.cfg.max_chunk_chars and group:
                flush()
            group.append(seg)
            size += len(seg_text)
        flush()


def _file_basename(path: str) -> str:
    return Path(path).name


def _text_chunks(
    path: str, text: str, kind: str, lang: str | None, cfg: Config, split_on_headings: bool
) -> list[Chunk]:
    lines = text.split("\n")
    out: list[Chunk] = []
    if split_on_headings:
        pat = re.compile(r"^(#{1,4}\s+\S|# %%)")
        bounds = [i for i, l in enumerate(lines) if pat.match(l)]
        if not bounds or bounds[0] != 0:
            bounds = [0] + bounds
        bounds.append(len(lines))
        for a, b in zip(bounds, bounds[1:]):
            sec = lines[a:b]
            if not "\n".join(sec).strip():
                continue
            cleaned_header = _HEADING_PREFIX_RE.sub("", sec[0]).strip()
            title = cleaned_header[:80] or _file_basename(path)
            for s, e, t in windows(sec, cfg.window_lines, cfg.window_overlap, offset=a):
                out.append(
                    Chunk(
                        id="", path=path, kind=kind, name=title, qualname=title,
                        start=s, end=e, language=lang, text=t,
                    )
                )
    else:
        fname = _file_basename(path)
        for s, e, t in windows(lines, cfg.window_lines, cfg.window_overlap):
            out.append(
                Chunk(
                    id="", path=path, kind=kind, name=fname, qualname=f"{fname}:{s}",
                    start=s, end=e, language=lang, text=t,
                )
            )
    return out


def chunk_file(
    path: str, content: str, kind: str, lang: str | None, cfg: Config
) -> list[Chunk]:
    """Suddivide un file in chunk semantici dotati di ID univoci."""
    chunks: list[Chunk] = []
    if kind == "code" and lang in LANG_SPECS:
        try:
            chunks = _CodeChunker(path, content, lang, cfg).run()
        except Exception as exc:
            logger.warning("Fallback su finestre per %s: %r", path, exc)
            chunks = []

    if not chunks:
        if path.lower().endswith(".ipynb"):
            content = render_notebook(content)
            chunks = _text_chunks(path, content, "doc", None, cfg, split_on_headings=True)
        elif kind == "doc":
            chunks = _text_chunks(path, content, "doc", None, cfg, split_on_headings=True)
        elif kind == "config":
            chunks = _text_chunks(path, content, "config", None, cfg, split_on_headings=False)
        else:
            chunks = _text_chunks(path, content, "text", lang, cfg, split_on_headings=False)

    seen: dict[str, int] = {}
    for c in chunks:
        base = f"{c.path}:{c.start}-{c.end}:{c.qualname}"
        seen[base] = seen.get(base, 0) + 1
        c.id = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return chunks