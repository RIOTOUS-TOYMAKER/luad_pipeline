from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .inference import PredictionResult


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    confusion_matrix: np.ndarray
    tn: int
    fp: int
    fn: int
    tp: int
    sensitivity: float
    specificity: float
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    classification_report_dict: dict
    classification_report_text: str


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def calculate_binary_metrics(
    prediction_result: PredictionResult,
    *,
    classes: tuple[str, ...],
    positive_class: str,
) -> BinaryMetrics:
    if len(classes) != 2:
        raise ValueError("This evaluation pipeline currently requires two classes.")
    if positive_class not in classes:
        raise ValueError(
            f"Positive class {positive_class!r} is absent from {classes}."
        )

    y_true = prediction_result.labels
    y_pred = prediction_result.predictions
    positive_index = classes.index(positive_class)
    negative_index = 1 - positive_index

    true_positive_mask = y_true == positive_index
    predicted_positive_mask = y_pred == positive_index
    tp = int(np.sum(true_positive_mask & predicted_positive_mask))
    fn = int(np.sum(true_positive_mask & ~predicted_positive_mask))
    fp = int(np.sum(~true_positive_mask & predicted_positive_mask))
    tn = int(np.sum(~true_positive_mask & ~predicted_positive_mask))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[negative_index, positive_index],
    )

    positive_probabilities = prediction_result.probabilities[:, positive_index]
    binary_true = (y_true == positive_index).astype(np.int64)
    if np.unique(binary_true).size != 2:
        raise ValueError(
            "ROC/AUC requires both the positive and negative classes in the test set."
        )

    fpr, tpr, thresholds = roc_curve(binary_true, positive_probabilities)
    auc_value = float(roc_auc_score(binary_true, positive_probabilities))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=list(classes),
        output_dict=False,
        zero_division=0,
    )

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        confusion_matrix=matrix,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        sensitivity=_safe_ratio(tp, tp + fn),
        specificity=_safe_ratio(tn, tn + fp),
        auc=auc_value,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        classification_report_dict=report_dict,
        classification_report_text=report_text,
    )
