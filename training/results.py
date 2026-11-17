from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


RESULT_COLUMNS = [
    "experiment",
    "architecture",
    "source_patch_size",
    "model_input_size",
    "stain",
    "reinhard",
    "seed",
    "fold",
    "n_splits",
    "stratify_mode",
    "split_source",
    "train_patches",
    "validation_patches",
    "train_patients",
    "validation_patients",
    "patient_overlap_count",
    "best_epoch_at_val_loss",
    "best_val_loss",
    "train_acc_at_best_val_loss",
    "val_acc_at_best_val_loss",
    "best_checkpoint_path",
    "best_state_dict_path",
]


def save_run_config(path: Path, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_training_result(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.reader(existing)
            existing_header = next(reader, [])
        if existing_header != RESULT_COLUMNS:
            raise ValueError(
                f"Existing result file {path} has an incompatible header. "
                "Move or rename it before running this refactored pipeline."
            )

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})


def save_best_network_state_dict(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Extract only net.* weights from the best Lightning checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    lightning_state = checkpoint.get("state_dict")
    if not isinstance(lightning_state, dict):
        raise ValueError(
            f"Checkpoint has no Lightning state_dict: {checkpoint_path}"
        )

    network_state = {
        key.removeprefix("net."): value
        for key, value in lightning_state.items()
        if key.startswith("net.")
    }
    if not network_state:
        raise ValueError(
            f"No network weights with prefix 'net.' were found in {checkpoint_path}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network_state, output_path)
