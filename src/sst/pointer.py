"""Pointer file utilities for tracking large assets alongside code in Git.

A pointer file is a small YAML file committed to Git that records the identity
of a large asset (data file, model artifact) stored outside the repository.
It requires no special tooling to read or verify — only standard Python and
a checksum utility.
"""

import hashlib
from pathlib import Path

import yaml


def write_pointer_file(
    asset_path: Path,
    source: str,
    git_commit: str = "unknown",
    description: str = "",
    repo_root: Path | None = None,
) -> Path:
    """Write a YAML pointer file for a large asset.

    Args:
        asset_path: Absolute path to the asset file on disk.
        source: Canonical URL from which the asset can be downloaded
            (e.g. a GitHub release or Zenodo link).
        git_commit: Full 40-character SHA of the commit associated with
            this version of the asset. Defaults to "unknown".
        description: Short human-readable description of the asset.
        repo_root: Root of the Git repository, used to compute the
            repo-relative ``path`` field. Defaults to ``Path.cwd()``.

    Returns:
        Path of the written ``.pointer.yaml`` file.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    asset_path = Path(asset_path).resolve()
    repo_root = Path(repo_root).resolve()

    md5 = _md5(asset_path)
    size = asset_path.stat().st_size
    relative_path = str(asset_path.relative_to(repo_root))

    data: dict = {"path": relative_path, "md5": md5, "size": size, "source": source}
    if description:
        data["description"] = description
    if git_commit and git_commit != "unknown":
        data["git_commit"] = git_commit

    pointer_path = Path(str(asset_path) + ".pointer.yaml")
    pointer_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return pointer_path


def _md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
