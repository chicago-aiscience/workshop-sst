"""Script to return an MD5 hash for a file.

python scripts/get_md5.py --file < /full/path/to/file.ext >
"""

import argparse
import hashlib
from pathlib import Path

def _md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, help="Path to a data file to retrieve a hash for")
    args = parser.parse_args()

    hash = _md5(args.file)
    print(f"Here is the file's MD5 hash: {hash}")

if __name__ == "__main__":
    main()