"""Command-line interface for running the SST ETL workflow."""

from pathlib import Path

import typer

from .io import load_enso, load_sst
from .plot import make_trend_plot
from .transform import join_on_month, metrics, tidy

app = typer.Typer(help="SST CLI")


@app.command("run")
def run(
    sst: Path = Path("data/sst_sample.csv"),
    enso: Path = Path("data/nino34_sample.csv"),
    out_dir: Path = Path("artifacts"),
    start: str = "2000-01",
) -> None:
    """Run the SST ETL workflow end-to-end.

    Parameters
    ----------
    sst : pathlib.Path, default="data/sst_sample.csv"
        Location of the SST CSV file to ingest.
    enso : pathlib.Path, default="data/nino34_sample.csv"
        Location of the ENSO index CSV file to ingest.
    out_dir : pathlib.Path, default="artifacts"
        Directory where generated summary artifacts are written.
    start : str, default="2000-01"
        Earliest date to retain after joining the SST and ENSO data. Parsed
        to a timestamp via :func:`pandas.to_datetime`.

    Returns
    -------
    None
        Writes a metrics CSV and trend plot to ``out_dir`` and prints their
        locations.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    sst_df = tidy(load_sst(sst), date_col="date", value_col="sst_c", roll=12)
    enso_df = tidy(load_enso(enso), date_col="date", value_col="nino34", roll=12)

    joined = join_on_month(sst_df, enso_df, start=start)

    summary = metrics(joined)
    (out_dir / "summary.csv").write_text(summary.to_csv(index=False))

    fig = make_trend_plot(joined)
    fig.savefig(out_dir / "trends.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {out_dir / 'summary.csv'} and {out_dir / 'trends.png'}")


if __name__ == "__main__":  # pragma: no cover
    app()
