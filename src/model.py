"""
Anomaly detection models for Calgary Water Quality monitoring.

Implements an ensemble of four detection methods -- Isolation Forest,
Local Outlier Factor, statistical thresholding, and z-score analysis --
and combines them into a unified anomaly score.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ===================================================================
# Individual anomaly detectors
# ===================================================================

class IsolationForestDetector:
    """Wrapper around sklearn Isolation Forest."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on *X* and return binary anomaly labels (1 = anomaly, 0 = normal).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary labels where 1 indicates an anomaly.
        """
        X_scaled = self.scaler.fit_transform(X)
        preds = self.model.fit_predict(X_scaled)
        # sklearn returns -1 for anomalies, 1 for inliers
        return (preds == -1).astype(int)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)


class LOFDetector:
    """Wrapper around sklearn Local Outlier Factor."""

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=False,
        )
        self.scaler = StandardScaler()

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return binary anomaly labels (1 = anomaly, 0 = normal)."""
        X_scaled = self.scaler.fit_transform(X)
        preds = self.model.fit_predict(X_scaled)
        return (preds == -1).astype(int)

    def negative_outlier_factors(self, X: np.ndarray) -> np.ndarray:
        """Return the negative outlier factor scores after fit_predict."""
        return self.model.negative_outlier_factor_


class StatisticalDetector:
    """Detect anomalies where values fall beyond mean +/- n standard deviations."""

    def __init__(self, n_std: float = 3.0):
        self.n_std = n_std
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Flag samples where any feature exceeds mean +/- n*std.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Binary anomaly labels.
        """
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0)
        # Avoid division by zero
        safe_std = np.where(self._std == 0, 1.0, self._std)
        z = np.abs((X - self._mean) / safe_std)
        # Anomaly if any feature exceeds threshold
        return (np.nanmax(z, axis=1) > self.n_std).astype(int)

    def max_z_per_sample(self, X: np.ndarray) -> np.ndarray:
        """Return the maximum absolute z-score across features for each sample."""
        if self._mean is None:
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0)
        safe_std = np.where(self._std == 0, 1.0, self._std)
        z = np.abs((X - self._mean) / safe_std)
        return np.nanmax(z, axis=1)


class ZScoreDetector:
    """Per-feature z-score based anomaly detection."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels based on z-score threshold."""
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        safe_std = np.where(std == 0, 1.0, std)
        z_scores = np.abs((X - mean) / safe_std)
        # Count how many features exceed threshold
        exceed_count = np.nansum(z_scores > self.threshold, axis=1)
        return (exceed_count > 0).astype(int)

    @staticmethod
    def compute_zscores(X: np.ndarray) -> np.ndarray:
        """Return the z-scores matrix."""
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        safe_std = np.where(std == 0, 1.0, std)
        return (X - mean) / safe_std


# ===================================================================
# Ensemble
# ===================================================================

class WaterQualityAnomalyDetector:
    """Ensemble anomaly detector combining four methods.

    The ensemble anomaly score is the mean of the binary predictions
    from each method, yielding a value in [0, 1].  A sample is
    classified as an anomaly when the ensemble score exceeds a given
    threshold (default 0.5, i.e., majority vote).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        stat_n_std: float = 3.0,
        zscore_threshold: float = 3.0,
        ensemble_threshold: float = 0.5,
    ):
        self.contamination = contamination
        self.ensemble_threshold = ensemble_threshold

        self.isolation_forest = IsolationForestDetector(contamination=contamination)
        self.lof = LOFDetector(contamination=contamination)
        self.statistical = StatisticalDetector(n_std=stat_n_std)
        self.zscore = ZScoreDetector(threshold=zscore_threshold)

        self._is_fitted = False

    # ----- helpers ---------------------------------------------------
    @staticmethod
    def _prepare_features(X: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with column medians so sklearn models work."""
        X = np.array(X, dtype=np.float64)
        # Replace inf
        X[~np.isfinite(X)] = np.nan
        # Impute NaN with column median
        col_median = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_median, inds[1])
        return X

    # ----- core API --------------------------------------------------
    def fit_predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Run all detectors and return per-method labels + ensemble score.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``isolation_forest``, ``lof``, ``statistical``,
            ``zscore``, ``ensemble_score``, ``anomaly``.
        """
        X_clean = self._prepare_features(X)

        if X_clean.shape[0] < 5:
            logger.warning("Too few samples (%d) for reliable anomaly detection.", X_clean.shape[0])
            n = X_clean.shape[0]
            return {
                "isolation_forest": np.zeros(n, dtype=int),
                "lof": np.zeros(n, dtype=int),
                "statistical": np.zeros(n, dtype=int),
                "zscore": np.zeros(n, dtype=int),
                "ensemble_score": np.zeros(n, dtype=float),
                "anomaly": np.zeros(n, dtype=int),
            }

        results: dict[str, np.ndarray] = {}
        results["isolation_forest"] = self.isolation_forest.fit_predict(X_clean)
        results["lof"] = self.lof.fit_predict(X_clean)
        results["statistical"] = self.statistical.fit_predict(X_clean)
        results["zscore"] = self.zscore.fit_predict(X_clean)

        # Ensemble: mean vote
        votes = np.column_stack([
            results["isolation_forest"],
            results["lof"],
            results["statistical"],
            results["zscore"],
        ])
        results["ensemble_score"] = votes.mean(axis=1)
        results["anomaly"] = (results["ensemble_score"] >= self.ensemble_threshold).astype(int)

        self._is_fitted = True
        return results

    # ----- evaluation ------------------------------------------------
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute precision, recall, and F1 when ground-truth labels exist.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels (1 = anomaly).
        y_pred : np.ndarray
            Predicted binary labels.

        Returns
        -------
        dict[str, float]
            Dictionary with precision, recall, f1, and full report.
        """
        metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }
        logger.info("Evaluation -- Precision: %.3f  Recall: %.3f  F1: %.3f",
                     metrics["precision"], metrics["recall"], metrics["f1"])
        return metrics

    # ----- anomaly summary -------------------------------------------
    @staticmethod
    def anomaly_summary(
        df: pd.DataFrame,
        results: dict[str, np.ndarray],
        parameters: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a human-readable anomaly report.

        Parameters
        ----------
        df : pd.DataFrame
            The wide-form dataframe that was used for detection.  Must
            contain ``sample_site`` and ``sample_date``.
        results : dict[str, np.ndarray]
            Output of :meth:`fit_predict`.
        parameters : list[str] or None
            Parameter columns to include in the summary.

        Returns
        -------
        pd.DataFrame
            One row per detected anomaly with site, date, parameter
            values, individual detector flags, and ensemble score.
        """
        summary = df.copy()
        summary["ensemble_score"] = results["ensemble_score"]
        summary["anomaly"] = results["anomaly"]
        summary["if_flag"] = results["isolation_forest"]
        summary["lof_flag"] = results["lof"]
        summary["stat_flag"] = results["statistical"]
        summary["zscore_flag"] = results["zscore"]

        anomalies = summary[summary["anomaly"] == 1].copy()

        if anomalies.empty:
            logger.info("No anomalies detected.")
            return anomalies

        # Sort by ensemble score descending
        anomalies.sort_values("ensemble_score", ascending=False, inplace=True)

        keep_cols = ["sample_site", "sample_date", "ensemble_score",
                     "if_flag", "lof_flag", "stat_flag", "zscore_flag"]
        if parameters:
            keep_cols.extend([p for p in parameters if p in anomalies.columns])

        available = [c for c in keep_cols if c in anomalies.columns]
        return anomalies[available].reset_index(drop=True)

    # ----- persistence -----------------------------------------------
    def save(self, filename: str = "water_quality_anomaly_model.joblib") -> Path:
        """Save the fitted ensemble model to disk.

        Parameters
        ----------
        filename : str
            Name of the file inside the ``models/`` directory.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = MODELS_DIR / filename
        joblib.dump(self, filepath)
        logger.info("Model saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filename: str = "water_quality_anomaly_model.joblib") -> "WaterQualityAnomalyDetector":
        """Load a previously saved ensemble model.

        Parameters
        ----------
        filename : str
            Name of the file inside the ``models/`` directory.

        Returns
        -------
        WaterQualityAnomalyDetector
            The loaded model instance.
        """
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)
        return model


# ===================================================================
# Convenience function
# ===================================================================

def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    ensemble_threshold: float = 0.5,
) -> tuple["WaterQualityAnomalyDetector", dict[str, np.ndarray], pd.DataFrame]:
    """One-call convenience: fit the ensemble and return results + summary.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form dataframe with ``sample_site``, ``sample_date``,
        and numeric feature columns.
    feature_cols : list[str]
        Names of the columns to use as features.
    contamination : float
        Expected proportion of anomalies.
    ensemble_threshold : float
        Minimum ensemble score to flag as anomaly.

    Returns
    -------
    tuple
        (detector, results_dict, anomaly_summary_df)
    """
    X = df[feature_cols].values
    detector = WaterQualityAnomalyDetector(
        contamination=contamination,
        ensemble_threshold=ensemble_threshold,
    )
    results = detector.fit_predict(X)
    summary = detector.anomaly_summary(df, results, parameters=feature_cols)
    return detector, results, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_and_prepare

    long_df, wide_df = load_and_prepare()

    # Select feature columns (raw parameter values only)
    exclude = {"sample_site", "sample_date", "year", "month"}
    feat_cols = [
        c for c in wide_df.columns
        if c not in exclude and wide_df[c].dtype in ("float64", "int64")
    ]

    detector, results, summary = detect_anomalies(wide_df, feat_cols)
    print(f"Total samples  : {len(wide_df)}")
    print(f"Anomalies found: {results['anomaly'].sum()}")
    print(summary.head(10))

    detector.save()
