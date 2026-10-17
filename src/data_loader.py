"""Steps 1-3: collect, import and load the dataset (+ add the custom row)."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("retailpulse")
EXPECTED = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
ALIASES = {"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID", "Customer_ID": "Customer ID"}


def _find_local_file() -> Path | None:
    for pattern in ("*.csv", "*.xlsx", "*.xls", "*.zip"):
        hits = sorted(config.DATA_RAW.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _download_from_kaggle() -> Path | None:
    """Download with kagglehub (needs internet; Kaggle login only for some datasets)."""
    try:
        import kagglehub

        path = Path(kagglehub.dataset_download(config.KAGGLE_HANDLE))
        log.info("Downloaded Kaggle dataset to %s", path)
        for pattern in ("*.csv", "*.xlsx", "*.xls"):
            hits = sorted(path.rglob(pattern))
            if hits:
                return hits[0]
    except Exception as exc:  # network blocked, no credentials, package missing ...
        log.warning("Kaggle download failed: %s", exc)
    return None


def _read_file(path: Path, years: str = "2010-2011") -> pd.DataFrame:
    if path.suffix == ".zip":
        out_dir = config.DATA_RAW / "unzipped"
        out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
        inner = _find_in(out_dir)
        return _read_file(inner, years)
    if path.suffix in (".xlsx", ".xls"):
        # Read ONLY the sheet(s) we need, one at a time. Loading every sheet at once
        # (sheet_name=None) holds ~1M rows in memory and fails on low-RAM machines.
        with pd.ExcelFile(path) as xf:
            names = xf.sheet_names
            log.info("Excel sheets found: %s", names)
            if years == "both" or len(names) == 1:
                picks = names
            else:
                picks = [n for n in names if years in n][:1] or [names[-1]]
            log.info("Reading sheet(s): %s (this can take a few minutes)", picks)
            frames = []
            for n in picks:
                frames.append(xf.parse(n))
                log.info("  sheet '%s': %s rows", n, f"{len(frames[-1]):,}")
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return pd.read_csv(path, encoding="unicode_escape")


def _find_in(folder: Path) -> Path:
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        hits = sorted(folder.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No data file inside {folder}")


def add_custom_row(df: pd.DataFrame) -> pd.DataFrame:
    """Add the assignment's extra record. Flagged so it can be traced through the pipeline."""
    df = df.copy()
    df["is_custom_row"] = False
    row = pd.DataFrame([{**config.CUSTOM_ROW, "is_custom_row": True}])
    row["InvoiceDate"] = pd.to_datetime(row["InvoiceDate"])
    return pd.concat([df, row], ignore_index=True)


def load_data(path: str | None = None, years: str = "2010-2011", allow_sample: bool = False) -> pd.DataFrame:
    """Load the raw dataset.

    Order: explicit path -> file in data/raw -> Kaggle download -> (optional) synthetic sample.
    """
    src = Path(path) if path else (_find_local_file() or _download_from_kaggle())
    if src is not None:
        log.info("Loading %s", src)
        df = _read_file(src, years)
        df = df.rename(columns=ALIASES)
    elif allow_sample:
        from .make_sample_data import make_sample

        log.warning("USING SYNTHETIC SAMPLE DATA - put the real file in data/raw/ for real results")
        df = make_sample()
    else:
        raise FileNotFoundError(
            "No dataset found. Put the Kaggle file in data/raw/ (csv, xlsx or zip), "
            "or run with --sample to test with synthetic data."
        )

    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}. Found: {list(df.columns)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = add_custom_row(df)
    log.info("Loaded %s rows x %s columns (incl. 1 custom row)", *df.shape)
    return df
