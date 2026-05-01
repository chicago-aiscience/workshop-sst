"""Download and verify an asset described by a pointer file.

Usage:
    python scripts/fetch_from_pointer.py data/sst_sample.csv.pointer.yaml
    python scripts/fetch_from_pointer.py runs/sst_enso/model.joblib.pointer.yaml --out-dir /tmp/models
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml


def _md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def fetch(pointer_path: Path, out_dir: Path | None = None) -> Path:
    """Download the asset described by a pointer file, verifying its MD5.

    Skips download if the file already exists and its hash matches.

    Args:
        pointer_path: Path to the ``.pointer.yaml`` file.
        out_dir: Directory to write the asset into. Defaults to the asset's
            repo-relative path resolved from the repo root (the grandparent
            of the ``scripts/`` directory).

    Returns:
        Path of the downloaded (or already-present) asset.

    Raises:
        SystemExit: On hash mismatch or download failure.
    """
    data = yaml.safe_load(pointer_path.read_text())
    source: str = data["source"]
    expected_md5: str = data["md5"]
    relative_path: str = data["path"]

    if out_dir is not None:
        dest = out_dir / Path(relative_path).name
    else:
        repo_root = Path(__file__).resolve().parent.parent
        dest = repo_root / relative_path

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and _md5(dest) == expected_md5:
        print(f"Already verified, skipping: {dest}")
        return dest

    print(f"Downloading {source}")
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=dest.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)

    try:
        urllib.request.urlretrieve(source, tmp_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    actual_md5 = _md5(tmp_path)
    if actual_md5 != expected_md5:
        tmp_path.unlink(missing_ok=True)
        print(
            f"Hash mismatch: expected {expected_md5}, got {actual_md5}",
            file=sys.stderr,
        )
        sys.exit(1)

    tmp_path.rename(dest)
    print(f"Saved to: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pointer", type=Path, help="Path to a .pointer.yaml file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()
    fetch(args.pointer, args.out_dir)


if __name__ == "__main__":
    main()
