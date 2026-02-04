from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    return float(thresholds[idx]), float(youden[idx])


def _evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision) if len(precision) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    default_preds = (y_prob >= 0.5).astype(int)
    default_cm = confusion_matrix(y_true, default_preds).ravel().tolist()
    opt_thresh, _ = _optimal_threshold(y_true, y_prob)
    opt_preds = (y_prob >= opt_thresh).astype(int)
    opt_cm = confusion_matrix(y_true, opt_preds).ravel().tolist()

    return {
        "model": name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "confusion_matrix_default": default_cm,
        "confusion_matrix_optimal": opt_cm,
        "optimal_threshold": float(opt_thresh),
    }


def _fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ais_features: list[str],
    pw_features: list[str],
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "ais_only": _fit_logistic_regression(train_df[ais_features], train_df["label_delay_risk"]),
        "pw_only": _fit_logistic_regression(train_df[pw_features], train_df["label_delay_risk"]),
        "fused": _fit_logistic_regression(train_df[ais_features + pw_features], train_df["label_delay_risk"]),
    }

    results: Dict[str, Dict[str, float]] = {}
    roc_data = {}
    pr_data = {}

    for name, model in models.items():
        if name == "ais_only":
            features = ais_features
        elif name == "pw_only":
            features = pw_features
        else:
            features = ais_features + pw_features
        y_prob = model.predict_proba(test_df[features])[:, 1]
        y_true = test_df["label_delay_risk"].to_numpy()
        results[name] = _evaluate_model(name, y_true, y_prob)
        roc_data[name] = roc_curve(y_true, y_prob)
        pr_data[name] = precision_recall_curve(y_true, y_prob)

    _plot_roc(roc_data, output_dir / "fig_roc.png")
    _plot_pr(pr_data, output_dir / "fig_pr.png")
    _plot_calibration(models["fused"], test_df, ais_features + pw_features, output_dir / "fig_calibration.png")

    metrics_path = output_dir / "metrics.json"
    metrics_table_path = output_dir / "metrics_table.csv"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    pd.DataFrame(results).T.to_csv(metrics_table_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)
    return results


def _plot_roc(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_pr(pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (precision, recall, _) in pr_data.items():
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_calibration(model: Pipeline, test_df: pd.DataFrame, features: list[str], path: Path) -> None:
    y_prob = model.predict_proba(test_df[features])[:, 1]
    y_true = test_df["label_delay_risk"].to_numpy()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve (Fused Model)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
