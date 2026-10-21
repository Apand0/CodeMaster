"""Fase 1: Clonazione ottimizzata (anche per monorepo), calcolo hash SHA-256 e filtro file."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from config import Config, logger

EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".scala": "scala",
    ".sh": "bash", ".lua": "lua", ".r": "r",
}
DOC_EXTS = {".md", ".rst", ".txt", ".ipynb"}
CONFIG_EXTS = {".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".sql", ".proto", ".xml"}
CONFIG_NAMES = {
    "Dockerfile", "Makefile", "CMakeLists.txt", "requirements.txt",
    "environment.yml", "package.json", "Cargo.toml", "pyproject.toml", "go.mod",
}


def repo_name_from_url(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return re.sub(r"\.git$", "", name) or "repo"


def clone_repo(source: str, cfg: Config) -> Path:
    """Clona un URL git (usando filtro blobless per grandi repo) o apre una cartella locale."""
    if os.path.isdir(source):
        return Path(source).resolve()
    dest = cfg.repos_dir / repo_name_from_url(source)
    if dest.exists():
        logger.info("Repository già presente in locale: %s", dest)
        return dest
    cfg.repos_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--quiet"]
    if cfg.blobless_clone:
        cmd.append("--filter=blob:none")
    if cfg.clone_depth:
        cmd += ["--depth", str(cfg.clone_depth)]
    cmd += [source, str(dest)]
    logger.info("Esecuzione: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(4096)
    except OSError:
        return True


def _git_files(root: Path) -> list[str] | None:
    """Usa `git ls-files` per rispettare automaticamente tutti i .gitignore della repo."""
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", errors="replace")
        return [p for p in out.split("\0") if p]
    except Exception:
        return None


def list_repo_files(root: Path, cfg: Config) -> list[str]:
    """Restituisce i percorsi relativi (POSIX '/') dei file di testo pertinenti."""
    candidates = _git_files(root)
    if candidates is None:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in cfg.skip_dirs and not d.startswith(".git")]
            for fn in filenames:
                candidates.append(os.path.relpath(os.path.join(dirpath, fn), root))

    skip_dirs_set = set(cfg.skip_dirs)
    try:  # la work_dir (indice, cache) non deve mai finire nell'indice stesso
        work_rel: str | None = cfg.work_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        work_rel = None
    files: list[str] = []
    for rel in candidates:
        rel = rel.replace("\\", "/")
        if work_rel and (rel == work_rel or rel.startswith(work_rel + "/")):
            continue
        parts = rel.split("/")
        if any(p in skip_dirs_set for p in parts[:-1]):
            continue
        if rel.lower().endswith(cfg.skip_suffixes):
            continue
        p = root / rel
        if not p.is_file() or p.is_symlink():
            continue
        try:
            st_size = p.stat().st_size
            if st_size > cfg.max_file_bytes or st_size == 0:
                continue
        except OSError:
            continue
        if _looks_binary(p):
            continue
        files.append(rel)
    return sorted(files)


def file_sha256(path: Path) -> str:
    """Calcola l'hash SHA-256 di un file per l'indicizzazione incrementale."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def file_kind(rel_path: str) -> str:
    """Restituisce 'code' | 'doc' | 'config' | 'other'."""
    p = Path(rel_path)
    ext = p.suffix.lower()
    if ext in EXT_TO_LANG:
        return "code"
    if ext in DOC_EXTS or p.name.upper().startswith(("README", "CONTRIBUTING", "ARCHITECTURE")):
        return "doc"
    if ext in CONFIG_EXTS or p.name in CONFIG_NAMES:
        return "config"
    return "other"


def language_of(rel_path: str) -> str | None:
    return EXT_TO_LANG.get(Path(rel_path).suffix.lower())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")