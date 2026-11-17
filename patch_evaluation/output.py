from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inference import PredictionResult
from .metrics import BinaryMetrics


RESULT_COLUMNS = [
    "experiment",
    "stain",
    "reinhard",
    "seed",
    "fold",
    "architecture",
    "source_patch_size",
    "model_input_size",
    "checkpoint",
    "test_dir",
    "test_patches",
    "negative_class",
    "positive_class",
    "TN",
    "FP",
    "FN",
    "TP",
    "sensitivity",
    "specificity",
    "AUC",
    "accuracy",
    "integrated_gradients",
    "ig_images",
    "ig_baseline",
    "ig_target",
    "ig_n_steps",
    "ig_internal_batch_size",
]


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_compatible(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def save_predictions(
    path: Path,
    prediction_result: PredictionResult,
    *,
    classes: tuple[str, ...],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"probability_{name}" for name in classes]
    fieldnames = [
        "sample_index",
        "filename",
        "path",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "correct",
        *probability_columns,
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_index in enumerate(prediction_result.sample_indices):
            true_index = int(prediction_result.labels[row_index])
            predicted_index = int(prediction_result.predictions[row_index])
            sample_path = Path(prediction_result.paths[row_index])
            row: dict[str, Any] = {
                "sample_index": int(sample_index),
                "filename": sample_path.name,
                "path": str(sample_path),
                "true_index": true_index,
                "true_label": classes[true_index],
                "predicted_index": predicted_index,
                "predicted_label": classes[predicted_index],
                "correct": true_index == predicted_index,
            }
            for class_index, column in enumerate(probability_columns):
                row[column] = float(
                    prediction_result.probabilities[row_index, class_index]
                )
            writer.writerow(row)


def save_confusion_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    *,
    display_classes: tuple[str, str],
) -> None:
    pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in display_classes],
        columns=[f"predicted_{name}" for name in display_classes],
    ).to_csv(path)


def save_confusion_matrix_plot(
    path: Path,
    matrix: np.ndarray,
    *,
    display_classes: tuple[str, str],
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=display_classes,
        yticklabels=display_classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def save_classification_report(
    csv_path: Path,
    text_path: Path,
    metrics: BinaryMetrics,
) -> None:
    pd.DataFrame(metrics.classification_report_dict).transpose().to_csv(csv_path)
    text_path.write_text(metrics.classification_report_text, encoding="utf-8")


def save_roc_outputs(
    csv_path: Path,
    figure_path: Path,
    metrics: BinaryMetrics,
    *,
    positive_class: str,
    dpi: int,
) -> None:
    pd.DataFrame(
        {
            "false_positive_rate": metrics.fpr,
            "true_positive_rate": metrics.tpr,
            "threshold": metrics.thresholds,
        }
    ).to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(metrics.fpr, metrics.tpr, label=f"AUC = {metrics.auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve (positive class: {positive_class})",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def metrics_to_dict(
    metrics: BinaryMetrics,
    *,
    classes: tuple[str, ...],
    positive_class: str,
) -> dict[str, Any]:
    positive_index = classes.index(positive_class)
    negative_class = classes[1 - positive_index]
    return {
        "classes": list(classes),
        "negative_class": negative_class,
        "positive_class": positive_class,
        "accuracy": metrics.accuracy,
        "AUC": metrics.auc,
        "sensitivity": metrics.sensitivity,
        "specificity": metrics.specificity,
        "TN": metrics.tn,
        "FP": metrics.fp,
        "FN": metrics.fn,
        "TP": metrics.tp,
        "confusion_matrix_order": [negative_class, positive_class],
        "confusion_matrix": metrics.confusion_matrix,
        "classification_report": metrics.classification_report_dict,
    }


def append_result(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()

    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != RESULT_COLUMNS:
            raise ValueError(
                f"Existing result file {path} has an incompatible header. "
                "Move or rename it before using this refactored evaluator."
            )

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})
