from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class PredictionResult:
    sample_indices: np.ndarray
    paths: tuple[str, ...]
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.size)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> PredictionResult:
    model.eval()

    indices: list[int] = []
    paths: list[str] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    with torch.inference_mode():
        for images, batch_labels, batch_paths, batch_indices in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with _autocast_context(device, amp):
                logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise RuntimeError(
                    "Model output must have shape (batch, num_classes); "
                    f"received {tuple(logits.shape)}."
                )

            batch_probabilities = torch.softmax(logits.float(), dim=1)
            batch_predictions = torch.argmax(batch_probabilities, dim=1)

            indices.extend(int(value) for value in batch_indices.tolist())
            paths.extend(str(value) for value in batch_paths)
            labels.append(batch_labels.cpu().numpy().astype(np.int64, copy=False))
            predictions.append(
                batch_predictions.cpu().numpy().astype(np.int64, copy=False)
            )
            probabilities.append(
                batch_probabilities.cpu().numpy().astype(np.float32, copy=False)
            )

    if not indices:
        raise ValueError("The test DataLoader produced no samples.")

    index_array = np.asarray(indices, dtype=np.int64)
    label_array = np.concatenate(labels)
    prediction_array = np.concatenate(predictions)
    probability_array = np.concatenate(probabilities)

    order = np.argsort(index_array)
    sorted_indices = index_array[order]
    if not np.array_equal(sorted_indices, np.arange(len(sorted_indices))):
        raise RuntimeError(
            "Test sample indices were missing or duplicated; deterministic "
            "per-patch output cannot be constructed."
        )

    sorted_paths = tuple(paths[int(position)] for position in order)
    return PredictionResult(
        sample_indices=sorted_indices,
        paths=sorted_paths,
        labels=label_array[order],
        predictions=prediction_array[order],
        probabilities=probability_array[order],
    )
