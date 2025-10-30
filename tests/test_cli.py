"""NumPy-style tests for the SST CLI interface."""

import pathlib
import subprocess
import sys


def test_cli_runs(tmp_path: pathlib.Path) -> None:
    """Run the CLI module via `python -m` and ensure artifacts are produced.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as the artifact output
        location for the CLI invocation.

    Notes
    -----
    Executes the CLI end-to-end with bundled sample data, asserting both
    outputs are created and the process exits successfully.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    out = tmp_path / "artifacts"
    out.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "sst.cli",
        "run",
        "--sst",
        str(data_dir / "sst_sample.csv"),
        "--enso",
        str(data_dir / "nino34_sample.csv"),
        "--out-dir",
        str(out),
        "--start",
        "2000-01",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    assert (out / "summary.csv").exists()
    assert (out / "trends.png").exists()
