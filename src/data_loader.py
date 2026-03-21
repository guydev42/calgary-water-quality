"""
Data loader for Calgary Water Quality Anomaly Detection System.

Fetches watershed water quality data from the Calgary Open Data portal
via the Socrata API, caches locally, and engineers features for
anomaly detection.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN = "data.calgary.ca"
DATASET_ID = "y8as-bmzj"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "watershed_water_quality.csv"

COLUMNS = [
    "sample_site",
    "numeric_result",
    "formatted_result",
    "result_units",
    "latitude_degrees",
    "longitude_degrees",
    "sample_date",
    "parameter",
    "site_key",
]

# Key parameters commonly monitored in watershed studies
KEY_PARAMETERS = [
    "pH",
    "Temperature",
    "Conductance",
    "Dissolved Oxygen",
    "Turbidity",
]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_water_quality_data(limit: int = 50_000, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch watershed water quality data from the Calgary Open Data portal.

    Parameters
    ----------
    limit : int
        Maximum number of records to retrieve from the API.
    force_refresh : bool
        If *True*, bypass the local CSV cache and re-download.

    Returns
    -------
    pd.DataFrame
        Raw water quality data.
    """
    if CACHE_FILE.exists() and not force_refresh:
        logger.info("Loading cached data from %s", CACHE_FILE)
        df = pd.read_csv(CACHE_FILE)
        return df

    logger.info("Fetching data from Calgary Open Data portal (dataset %s)...", DATASET_ID)
    try:
        client = Socrata(DOMAIN, app_token=None, timeout=60)
        records = client.get(DATASET_ID, limit=limit)
        client.close()

        df = pd.DataFrame.from_records(records)

        # Keep only the columns we need (others may exist in the dataset)
        available = [c for c in COLUMNS if c in df.columns]
        df = df[available]

        # Ensure data directory exists and cache
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)
        logger.info("Fetched and cached %d records to %s", len(df), CACHE_FILE)
    except Exception as exc:
        logger.error("Failed to fetch data from Socrata API: %s", exc)
        if CACHE_FILE.exists():
            logger.warning("Falling back to cached data.")
            return pd.read_csv(CACHE_FILE)
        raise

    return df


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the raw water quality dataframe.

    Steps
    -----
    1. Parse ``sample_date`` to datetime.
    2. Convert ``numeric_result`` to float.
    3. Extract year and month columns.
    4. Drop rows without a valid date or numeric result.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Parse dates
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df.dropna(subset=["sample_date"], inplace=True)

    # Numeric result
    df["numeric_result"] = pd.to_numeric(df["numeric_result"], errors="coerce")

    # Time features
    df["year"] = df["sample_date"].dt.year
    df["month"] = df["sample_date"].dt.month

    # Sort chronologically
    df.sort_values(["sample_site", "sample_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Coordinates
    for col in ("latitude_degrees", "longitude_degrees"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Pivot (wide-form)
# ---------------------------------------------------------------------------
def pivot_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot so each parameter becomes its own column per site and date.

    The resulting dataframe has one row per (sample_site, sample_date)
    combination, with columns named after the water quality parameters
    (e.g., ``pH``, ``Temperature``).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed long-form data (must contain ``sample_site``,
        ``sample_date``, ``parameter``, ``numeric_result``).

    Returns
    -------
    pd.DataFrame
        Wide-form dataframe.
    """
    required = {"sample_site", "sample_date", "parameter", "numeric_result"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    pivot = df.pivot_table(
        index=["sample_site", "sample_date"],
        columns="parameter",
        values="numeric_result",
        aggfunc="mean",
    )
    pivot.columns.name = None
    pivot.reset_index(inplace=True)

    return pivot


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_rolling_statistics(
    df: pd.DataFrame,
    parameters: list[str] | None = None,
    windows: tuple[int, ...] = (7, 30),
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features per site.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form dataframe produced by :func:`pivot_parameters`.
    parameters : list[str] or None
        Parameter columns to compute rolling stats for.  If *None*,
        auto-detect numeric columns other than metadata.
    windows : tuple[int, ...]
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame
        Dataframe with new rolling-statistic columns.
    """
    df = df.copy()
    if parameters is None:
        exclude = {"sample_site", "sample_date", "year", "month"}
        parameters = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]

    df.sort_values(["sample_site", "sample_date"], inplace=True)

    for site in df["sample_site"].unique():
        mask = df["sample_site"] == site
        for param in parameters:
            if param not in df.columns:
                continue
            series = df.loc[mask, param]
            for w in windows:
                df.loc[mask, f"{param}_roll_mean_{w}d"] = series.rolling(window=w, min_periods=1).mean()
                df.loc[mask, f"{param}_roll_std_{w}d"] = series.rolling(window=w, min_periods=1).std()

    return df


def add_rate_of_change(df: pd.DataFrame, parameters: list[str] | None = None) -> pd.DataFrame:
    """Calculate rate-of-change (first difference) features per site.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form dataframe.
    parameters : list[str] or None
        Parameter columns.  Auto-detected when *None*.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``<param>_roc`` columns appended.
    """
    df = df.copy()
    if parameters is None:
        exclude = {"sample_site", "sample_date", "year", "month"}
        parameters = [
            c for c in df.columns
            if c not in exclude
            and not c.endswith(("_roll_mean_7d", "_roll_std_7d", "_roll_mean_30d", "_roll_std_30d"))
            and df[c].dtype in ("float64", "int64")
        ]

    df.sort_values(["sample_site", "sample_date"], inplace=True)
    for site in df["sample_site"].unique():
        mask = df["sample_site"] == site
        for param in parameters:
            if param not in df.columns:
                continue
            df.loc[mask, f"{param}_roc"] = df.loc[mask, param].diff()

    return df


def add_zscore_features(df: pd.DataFrame, parameters: list[str] | None = None) -> pd.DataFrame:
    """Compute z-score features for each parameter within each site.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form dataframe.
    parameters : list[str] or None
        Parameter columns.  Auto-detected when *None*.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``<param>_zscore`` columns appended.
    """
    df = df.copy()
    if parameters is None:
        exclude = {"sample_site", "sample_date", "year", "month"}
        parameters = [
            c for c in df.columns
            if c not in exclude
            and "_roll_" not in c
            and "_roc" not in c
            and "_zscore" not in c
            and df[c].dtype in ("float64", "int64")
        ]

    for site in df["sample_site"].unique():
        mask = df["sample_site"] == site
        for param in parameters:
            if param not in df.columns:
                continue
            series = df.loc[mask, param]
            mean = series.mean()
            std = series.std()
            if std and std > 0:
                df.loc[mask, f"{param}_zscore"] = (series - mean) / std
            else:
                df.loc[mask, f"{param}_zscore"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def load_and_prepare(limit: int = 50_000, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full data pipeline: fetch, preprocess, pivot, engineer.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (long_df, wide_df) where *long_df* is the cleaned
        long-form data and *wide_df* is the pivoted, feature-enriched
        dataframe ready for modelling.
    """
    raw = fetch_water_quality_data(limit=limit, force_refresh=force_refresh)
    long_df = preprocess(raw)

    wide_df = pivot_parameters(long_df)
    wide_df = add_rolling_statistics(wide_df)
    wide_df = add_rate_of_change(wide_df)
    wide_df = add_zscore_features(wide_df)

    return long_df, wide_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    long_df, wide_df = load_and_prepare()
    print(f"Long-form shape : {long_df.shape}")
    print(f"Wide-form shape : {wide_df.shape}")
    print(f"Sites           : {long_df['sample_site'].nunique()}")
    print(f"Parameters      : {long_df['parameter'].nunique()}")
    print(f"Date range      : {long_df['sample_date'].min()} to {long_df['sample_date'].max()}")
