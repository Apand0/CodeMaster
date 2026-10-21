"""Fase 3: Indice Ibrido Incrementale (SHA-256) scalabile su grandi repo, con Pre-Filtering e Outline."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from chunker import Chunk, chunk_file
from config import Config, get_secret, logger
from ingest import file_kind, file_sha256, language_of, list_repo_files, read_text

SYMBOL_KINDS = {"function", "method", "class", "struct", "interface", "enum", "impl", "trait"}


def _norm_rel(p: str) -> str:
    """Normalizza un percorso relativo senza mangiare i punti iniziali (es. '.github/x')."""
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


# --------------------------------------------------------------------------------------
# Embedders (Locale CPU/GPU, Cloud API o Fast Hashing per Monorepo)
# --------------------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Modello di embedding da Hugging Face (leggero, gira anche su CPU)."""

    def __init__(self, cfg: Config, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedding_model, device=device)
        self.model.max_seq_length = 512
        self.name = cfg.embedding_model

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.cfg.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )

    def encode_query(self, text: str) -> np.ndarray:
        prefix = self.cfg.query_instruction if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0]


class GeminiAPIEmbedder:
    """Embedder basato sulle API gratuite di Google Gemini (zero carico sulla CPU locale)."""

    def __init__(self, cfg: Config, api_key: str | None = None) -> None:
        import litellm
        self._litellm = litellm
        self.cfg = cfg
        self.api_key = api_key or get_secret("GEMINI_API_KEY", prompt_if_missing=True)
        dims = cfg.gemini_embedding_dims
        self.dims = dims
        # le dimensioni stanno nel nome: cambiandole l'indice viene ricostruito invece di andare in conflitto
        self.name = f"gemini_api:{cfg.gemini_embedding_model}" + (f":{dims}" if dims else "")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model_name = self.cfg.gemini_embedding_model
        if not model_name.startswith("gemini/"):
            model_name = "gemini/" + model_name.replace("models/", "")
        extra = {"dimensions": self.dims} if self.dims else {}
        resp = self._litellm.embedding(model=model_name, input=texts, api_key=self.api_key, **extra)
        vecs = np.array([item["embedding"] for item in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        out = []
        bs = min(self.cfg.embed_batch_size, 64)
        for i in range(0, len(texts), bs):
            out.append(self._embed_batch(texts[i : i + bs]))
        return np.vstack(out) if out else np.empty((0, self.dims or 3072), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]


class HashingEmbedder:
    """Embedder ultra-veloce su CPU senza download (ideale per monorepo enormi su portatili)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = "hashing-embedder"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.empty((0, self.dim), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def select_embedder(cfg: Config, num_chunks: int, saved_model_name: str | None = None) -> Any:
    """Seleziona automaticamente l'embedder ottimale in base alla configurazione e alla scala della repo."""
    if saved_model_name == "hashing-embedder" or cfg.embedder_type == "hashing":
        return HashingEmbedder()
    if (saved_model_name and saved_model_name.startswith("gemini_api:")) or cfg.embedder_type == "gemini_api":
        return GeminiAPIEmbedder(cfg)
    if cfg.embedder_type == "auto" and num_chunks > cfg.auto_fast_embed_threshold_chunks:
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False
        if not has_gpu:
            logger.warning(
                "Repo molto grande (%d chunk) rilevata su CPU: uso HashingEmbedder + BM25 per indicizzazione istantanea.",
                num_chunks,
            )
            return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(cfg)
    except Exception as exc:
        logger.warning("Impossibile caricare SentenceTransformer (%s), fallback su HashingEmbedder.", exc)
        return HashingEmbedder()


# --------------------------------------------------------------------------------------
# Tokenizzazione per BM25
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for w in _WORD.findall(text):
        low = w.lower()
        toks.append(low)
        parts = [p.lower() for p in _CAMEL.findall(w.replace("_", " "))]
        if len(parts) > 1:
            toks.extend(parts)
    return toks


def embed_text(c: Chunk, cfg: Config) -> str:
    imports_str = f"imports: {', '.join(c.imports[:10])}\n" if c.imports else ""
    calls_str = f"calls: {', '.join(c.calls[:15])}\n" if c.calls else ""
    head = f"{c.path} | {c.kind} {c.qualname}\n{c.signature}\n{imports_str}{calls_str}{c.docstring}\n"
    return (head + c.text)[: cfg.embed_max_chars]


def bm25_text(c: Chunk) -> str:
    return f"{c.path} {c.qualname} {c.signature} {' '.join(c.calls)} {c.docstring} {c.text}"


# --------------------------------------------------------------------------------------
# RepoIndex (con aggiornamento incrementale SHA-256)
# --------------------------------------------------------------------------------------


class RepoIndex:
    def __init__(
        self,
        repo_root: Path,
        cfg: Config,
        chunks: list[Chunk],
        files: list[str],
        embedder: Any = None,
        collection: Any = None,
        file_hashes: dict[str, str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.cfg = cfg
        self.chunks = chunks
        self.files = files
        self.embedder = embedder
        self.collection = collection
        self.file_hashes = file_hashes or {}
        self.by_id: dict[str, Chunk] = {c.id: c for c in chunks}
        self._build_lookups()
        self._build_bm25()

    @staticmethod
    def index_path(cfg: Config, repo_root: Path) -> Path:
        return cfg.index_dir / Path(repo_root).name

    @classmethod
    def build(
        cls, repo_root: Path, cfg: Config, embedder: Any = None, force_full: bool = False
    ) -> RepoIndex:
        """Costruisce o aggiorna incrementalmente l'indice calcolando gli SHA-256 dei file."""
        repo_root = Path(repo_root).resolve()
        out_dir = cls.index_path(cfg, repo_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        files = list_repo_files(repo_root, cfg)
        current_hashes: dict[str, str] = {}
        for rel in files:
            try:
                current_hashes[rel] = file_sha256(repo_root / rel)
            except OSError:
                continue

        old_hashes: dict[str, str] = {}
        old_chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
        saved_embed_model: str | None = None

        meta_file = out_dir / "meta.json"
        chunks_file = out_dir / "chunks.json"
        if not force_full and meta_file.exists() and chunks_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                old_hashes = meta.get("file_hashes", {})
                saved_embed_model = meta.get("embedding_model")
                for d in json.loads(chunks_file.read_text(encoding="utf-8")):
                    c = Chunk.from_dict(d)
                    old_chunks_by_file[c.path].append(c)
            except Exception as exc:
                logger.warning("Indice precedente non leggibile (%s), ricostruzione completa.", exc)
                old_hashes = {}
                old_chunks_by_file.clear()

        added_or_changed = [f for f in files if old_hashes.get(f) != current_hashes.get(f)]
        removed_files = [f for f in old_hashes if f not in current_hashes]

        logger.info(
            "Analisi %s: %d file totali (%d nuovi/modificati, %d rimossi)",
            repo_root.name, len(files), len(added_or_changed), len(removed_files),
        )

        new_chunks_by_file: dict[str, list[Chunk]] = {}
        for rel in added_or_changed:
            try:
                content = read_text(repo_root / rel)
            except OSError:
                continue
            new_chunks_by_file[rel] = chunk_file(rel, content, file_kind(rel), language_of(rel), cfg)

        all_chunks: list[Chunk] = []
        for rel in files:
            if rel in new_chunks_by_file:
                all_chunks.extend(new_chunks_by_file[rel])
            elif rel in old_chunks_by_file:
                all_chunks.extend(old_chunks_by_file[rel])

        embedder = embedder or select_embedder(cfg, len(all_chunks), saved_embed_model)

        import chromadb
        client = chromadb.PersistentClient(path=str(out_dir / "chroma"))

        # Se l'embedder è cambiato o è richiesto force_full, ricrea la collezione da zero
        rebuild_vectors = (
            force_full
            or not old_hashes
            or (saved_embed_model is not None and saved_embed_model != getattr(embedder, "name", "?"))
        )

        if rebuild_vectors:
            try:
                client.delete_collection("chunks")
            except Exception:
                pass
            col = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
            chunks_to_embed = all_chunks
        else:
            col = client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
            # Rimuove i vecchi vettori dei file modificati o cancellati
            ids_to_delete = [
                c.id
                for f in (added_or_changed + removed_files)
                for c in old_chunks_by_file.get(f, [])
            ]
            for i in range(0, len(ids_to_delete), 500):
                col.delete(ids=ids_to_delete[i : i + 500])
            chunks_to_embed = [c for f in added_or_changed for c in new_chunks_by_file.get(f, [])]

        batch_size = 500
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i : i + batch_size]
            vecs = embedder.encode_docs([embed_text(c, cfg) for c in batch])
            col.upsert(
                ids=[c.id for c in batch],
                embeddings=[v.tolist() for v in vecs],
                metadatas=[
                    {
                        "path": c.path,
                        "kind": c.kind,
                        "qualname": c.qualname,
                        "start": c.start,
                        "end": c.end,
                    }
                    for c in batch
                ],
            )
            logger.info("Embedding indicizzati: %d/%d", min(i + batch_size, len(chunks_to_embed)), len(chunks_to_embed))

        chunks_file.write_text(json.dumps([c.to_dict() for c in all_chunks]), encoding="utf-8")
        meta_file.write_text(
            json.dumps({
                "repo_root": str(repo_root),
                "files": files,
                "file_hashes": current_hashes,
                "embedding_model": getattr(embedder, "name", "?"),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }),
            encoding="utf-8",
        )
        logger.info("Indice pronto in %.1fs -> %s", time.time() - t0, out_dir)
        return cls(repo_root, cfg, all_chunks, files, embedder, col, current_hashes)

    @classmethod
    def load(cls, repo_root: Path, cfg: Config, embedder: Any = None) -> RepoIndex:
        """Carica l'indice aggiornando automaticamente eventuali file cambiati."""
        repo_root = Path(repo_root).resolve()
        out_dir = cls.index_path(cfg, repo_root)
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk.from_dict(d)
            for d in json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
        ]
        import chromadb
        col = chromadb.PersistentClient(path=str(out_dir / "chroma")).get_collection("chunks")
        if embedder is None:
            embedder = select_embedder(cfg, len(chunks), meta.get("embedding_model"))
        return cls(repo_root, cfg, chunks, meta["files"], embedder, col, meta.get("file_hashes", {}))

    @classmethod
    def exists(cls, repo_root: Path, cfg: Config) -> bool:
        return (cls.index_path(cfg, Path(repo_root).resolve()) / "meta.json").exists()

    def _build_lookups(self) -> None:
        self.symbols: dict[str, list[Chunk]] = defaultdict(list)
        self.callers: dict[str, list[Chunk]] = defaultdict(list)
        self.by_file: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            self.by_file[c.path].append(c)
            if c.kind in SYMBOL_KINDS:
                self.symbols[c.name].append(c)
            if c.kind in ("function", "method") and ("[parte " not in c.qualname or "[parte 1/" in c.qualname):
                for callee in c.calls:
                    self.callers[callee].append(c)

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        self._bm25 = (
            BM25Okapi([tokenize(bm25_text(c)) or ["_"] for c in self.chunks])
            if self.chunks
            else None
        )

    def find_symbol(self, name: str) -> list[Chunk]:
        name = name.strip()
        short = name.split(".")[-1]
        exact = [
            c for c in self.symbols.get(short, [])
            if c.qualname.split(" [")[0] == name
            or c.qualname.split(" [")[0].endswith("." + name)
            or c.name == name
        ]
        if exact:
            return sorted(exact, key=lambda c: (c.path, c.start))
        low = name.lower()
        partial = [c for cs in self.symbols.values() for c in cs if low in c.qualname.lower()]
        return sorted(partial, key=lambda c: (c.path, c.start))[:25]

    def get_file_outline(self, rel_path: str) -> str:
        """Genera lo scheletro strutturale completo di un file (import, classi, firme, chiamate)."""
        norm_path = _norm_rel(rel_path)
        if norm_path not in self.by_file:
            matches = [f for f in self.files if f.endswith(norm_path)]
            if len(matches) == 1:
                norm_path = matches[0]
            elif len(matches) > 1:
                return f"Percorso ambiguo '{rel_path}'. Corrispondenze: {', '.join(matches[:10])}"
            else:
                return f"File '{rel_path}' non trovato nell'indice."

        f_chunks = sorted(self.by_file[norm_path], key=lambda c: c.start)
        lines: list[str] = [f"=== Struttura AST del file: {norm_path} ==="]

        imports = next((c.imports for c in f_chunks if c.imports), [])
        if imports:
            lines.append("\n[Import e Dipendenze]")
            lines.extend(f"  {imp}" for imp in imports[:30])

        lines.append("\n[Simboli e Definizioni]")
        for c in f_chunks:
            if "[parte " in c.qualname and "[parte 1/" not in c.qualname:
                continue
            sig_info = f" -> {c.signature}" if c.signature else ""
            doc_info = f"\n      doc: {c.docstring.splitlines()[0][:120]}" if c.docstring else ""
            calls_info = f"\n      chiama: {', '.join(c.calls[:12])}" if c.calls else ""
            lines.append(
                f"  • righe {c.start:>4}-{c.end:<4} | [{c.kind}] {c.qualname}{sig_info}{doc_info}{calls_info}"
            )
        return "\n".join(lines)

    def search(
        self, query: str, k: int = 6, path_prefix: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Ricerca ibrida (Semantica + BM25) con Pre-Filtering sui percorsi per repo di ogni taglia."""
        if not self.chunks:
            return []

        norm_prefix = _norm_rel(path_prefix) if path_prefix else None
        valid_indices = [
            idx for idx, c in enumerate(self.chunks)
            if not norm_prefix or c.path.startswith(norm_prefix)
        ]
        if not valid_indices:
            return []

        pool = max(k * 5, 30)
        scores: dict[str, float] = defaultdict(float)

        # 1) Ricerca semantica
        if self.collection is not None and self.embedder is not None:
            qv = self.embedder.encode_query(query).tolist()
            # Se norm_prefix è esattamente un file, usa il filtro nativo di ChromaDB
            where_filter = {"path": norm_prefix} if (norm_prefix and norm_prefix in self.by_file) else None
            n_query = min(len(valid_indices) if where_filter else len(self.chunks), pool * 4 if norm_prefix else pool)
            if n_query > 0:
                res = self.collection.query(
                    query_embeddings=[qv],
                    n_results=n_query,
                    where=where_filter,
                )
                sem_rank = 0
                for cid in res["ids"][0]:
                    c = self.by_id.get(cid)
                    if c is None or (norm_prefix and not c.path.startswith(norm_prefix)):
                        continue
                    scores[cid] += 1.0 / (60 + sem_rank)
                    sem_rank += 1
                    if sem_rank >= pool:
                        break

        # 2) Ricerca lessicale BM25 pre-filtrata
        if self._bm25 is not None:
            bm_all = self._bm25.get_scores(tokenize(query) or ["_"])
            valid_sorted = sorted(valid_indices, key=lambda idx: -bm_all[idx])[:pool]
            for rank, idx in enumerate(valid_sorted):
                if bm_all[idx] > 0:
                    scores[self.chunks[idx].id] += 1.0 / (60 + rank)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.by_id[cid], sc) for cid, sc in ranked]

    def build_repo_map(self, max_chars: int | None = None) -> str:
        max_chars = max_chars or self.cfg.repo_map_chars
        by_file_syms: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            if c.kind in SYMBOL_KINDS and ("[parte " not in c.qualname or "[parte 1/" in c.qualname):
                by_file_syms[c.path].append(c)

        def level2() -> str:
            lines = []
            for f in self.files:
                syms = by_file_syms.get(f, [])
                if not syms:
                    lines.append(f)
                    continue
                classes = [c for c in syms if c.kind not in ("function", "method")]
                funcs = [c for c in syms if c.kind == "function"]
                parts = []
                for cl in classes:
                    base_qual = cl.qualname.split(" [")[0]
                    meths = [
                        m.name for m in syms
                        if m.kind == "method" and m.qualname.startswith(base_qual + ".")
                    ]
                    m_str = f"[{', '.join(meths[:8])}{'...' if len(meths) > 8 else ''}]" if meths else ""
                    parts.append(f"{cl.kind} {cl.name}{m_str}")
                parts += [f"def {fn.name}" for fn in funcs[:8]]
                if len(funcs) > 8:
                    parts.append("...")
                lines.append(f"{f}: " + "; ".join(parts))
            return "\n".join(lines)

        def level1() -> str:
            return "\n".join(self.files)

        def level0() -> str:
            counts: dict[str, int] = defaultdict(int)
            for f in self.files:
                parts = f.split("/")
                key = (
                    "/".join(parts[:2]) + "/"
                    if len(parts) > 2
                    else ("/".join(parts[:1]) + "/" if len(parts) > 1 else "(root)")
                )
                counts[key] += 1
            return "\n".join(f"{d} ({n} file)" for d, n in sorted(counts.items()))

        for fn in (level2, level1, level0):
            txt = fn()
            if len(txt) <= max_chars:
                return txt
        return level0()[:max_chars] + "\n..."

    def stats(self) -> str:
        langs: dict[str, int] = defaultdict(int)
        for f in self.files:
            langs[language_of(f) or file_kind(f)] += 1
        top = ", ".join(f"{k}: {v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:6])
        return f"{len(self.files)} file, {len(self.chunks)} chunk, {len(self.symbols)} simboli ({top})"