from __future__ import annotations

import csv
import json
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .metadata import PatchSample


@dataclass(frozen=True)
class SplitIndices:
    train_indices: np.ndarray
    val_indices: np.ndarray
    seed: int
    fold: int
    n_splits: int
    stratify_mode: str
    source: str = "StratifiedGroupKFold"


def _stratification_labels(
    samples: Sequence[PatchSample],
    mode: str,
) -> np.ndarray:
    if mode == "binary":
        return np.asarray([sample.binary_stratum for sample in samples])
    if mode == "binary_subtype":
        return np.asarray([sample.binary_subtype_stratum for sample in samples])
    raise ValueError(f"Unsupported stratify mode: {mode!r}")


def _warn_about_limited_strata(
    samples: Sequence[PatchSample],
    labels: np.ndarray,
    n_splits: int,
) -> None:
    groups_by_stratum: dict[str, set[str]] = defaultdict(set)
    for sample, stratum in zip(samples, labels, strict=True):
        groups_by_stratum[str(stratum)].add(sample.patient_id)

    limited = {
        stratum: len(groups)
        for stratum, groups in groups_by_stratum.items()
        if len(groups) < n_splits
    }
    if limited:
        detail = ", ".join(
            f"{stratum}={count} patient(s)"
            for stratum, count in sorted(limited.items())
        )
        warnings.warn(
            "Some strata occur in fewer patient groups than n_splits; exact "
            f"stratification across all folds is impossible: {detail}",
            stacklevel=2,
        )


def _validate_split(
    samples: Sequence[PatchSample],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> None:
    train_patients = {samples[int(i)].patient_id for i in train_indices}
    val_patients = {samples[int(i)].patient_id for i in val_indices}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(
            "Patient leakage detected after splitting: "
            + ", ".join(sorted(overlap))
        )

    for split_name, indices in (
        ("training", train_indices),
        ("validation", val_indices),
    ):
        labels = {samples[int(i)].label for i in indices}
        if labels != {0, 1}:
            raise RuntimeError(
                f"The {split_name} split does not contain both binary classes; "
                f"labels present: {sorted(labels)}."
            )


def create_stratified_group_split(
    samples: Sequence[PatchSample],
    *,
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 1,
    stratify_mode: str = "binary_subtype",
) -> SplitIndices:
    if not samples:
        raise ValueError("samples must not be empty.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}].")

    unique_patients = {sample.patient_id for sample in samples}
    if len(unique_patients) < n_splits:
        raise ValueError(
            f"At least {n_splits} distinct patients are required, but only "
            f"{len(unique_patients)} were found."
        )

    y = _stratification_labels(samples, stratify_mode)
    groups = np.asarray([sample.patient_id for sample in samples])
    X = np.zeros((len(samples), 1), dtype=np.uint8)

    _warn_about_limited_strata(samples, y, n_splits)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = list(splitter.split(X=X, y=y, groups=groups))
    train_indices, val_indices = folds[fold]

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    _validate_split(samples, train_indices, val_indices)

    assigned = np.concatenate((train_indices, val_indices))
    if assigned.size != len(samples) or np.unique(assigned).size != len(samples):
        raise RuntimeError(
            "The generated split does not assign every patch exactly once."
        )

    return SplitIndices(
        train_indices=train_indices,
        val_indices=val_indices,
        seed=seed,
        fold=fold,
        n_splits=n_splits,
        stratify_mode=stratify_mode,
    )


def _patient_assignments(
    samples: Sequence[PatchSample],
    split: SplitIndices,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for index in split.train_indices:
        assignments[samples[int(index)].patient_id] = "train"
    for index in split.val_indices:
        patient_id = samples[int(index)].patient_id
        previous = assignments.get(patient_id)
        if previous is not None and previous != "validation":
            raise RuntimeError(f"Patient {patient_id} is assigned to both splits.")
        assignments[patient_id] = "validation"
    return assignments


def save_patient_split(
    path: Path,
    samples: Sequence[PatchSample],
    split: SplitIndices,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments = _patient_assignments(samples, split)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patient_id",
                "split",
                "seed",
                "fold",
                "n_splits",
                "stratify_mode",
            ]
        )
        for patient_id, split_name in sorted(assignments.items()):
            writer.writerow(
                [
                    patient_id,
                    split_name,
                    split.seed,
                    split.fold,
                    split.n_splits,
                    split.stratify_mode,
                ]
            )


def _load_patient_assignments(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    assignments: dict[str, str] = {}
    metadata: dict[str, str] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"patient_id", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {sorted(required)}."
            )

        for row in reader:
            patient_id = row["patient_id"].strip().lower()
            split_name = row["split"].strip().lower()
            if split_name not in {"train", "validation"}:
                raise ValueError(
                    f"Invalid split {split_name!r} for patient {patient_id}."
                )
            previous = assignments.get(patient_id)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Patient {patient_id} has conflicting assignments in {path}."
                )
            assignments[patient_id] = split_name

            for key in ("seed", "fold", "n_splits", "stratify_mode"):
                value = (row.get(key) or "").strip()
                if value and key not in metadata:
                    metadata[key] = value

    return assignments, metadata


def _apply_patient_assignments(
    samples: Sequence[PatchSample],
    assignments: dict[str, str],
    *,
    seed: int,
    fold: int,
    n_splits: int,
    stratify_mode: str,
    source: str,
) -> SplitIndices:
    sample_patients = {sample.patient_id for sample in samples}
    missing = sorted(sample_patients - set(assignments))
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise ValueError(
            "The existing split file has no assignment for these patients: "
            f"{preview}{suffix}"
        )

    train_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if assignments[sample.patient_id] == "train"
        ],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if assignments[sample.patient_id] == "validation"
        ],
        dtype=np.int64,
    )

    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("The loaded split produced an empty train or validation set.")

    _validate_split(samples, train_indices, val_indices)
    return SplitIndices(
        train_indices=train_indices,
        val_indices=val_indices,
        seed=seed,
        fold=fold,
        n_splits=n_splits,
        stratify_mode=stratify_mode,
        source=source,
    )


def resolve_split(
    samples: Sequence[PatchSample],
    *,
    n_splits: int,
    fold: int,
    seed: int,
    stratify_mode: str,
    split_file: Path | None = None,
) -> SplitIndices:
    """
    Load an existing patient-level split or create one with SGKF.

    When split_file is supplied but does not yet exist, a new split is created
    and saved there. Reusing that file for other stains gives identical patient
    assignments across stains.
    """
    if split_file is not None:
        split_file = Path(split_file)
        if split_file.exists():
            assignments, metadata = _load_patient_assignments(split_file)

            expected = {
                "seed": str(seed),
                "fold": str(fold),
                "n_splits": str(n_splits),
                "stratify_mode": stratify_mode,
            }
            conflicts = {
                key: (metadata[key], expected[key])
                for key in expected
                if key in metadata and metadata[key] != expected[key]
            }
            if conflicts:
                detail = ", ".join(
                    f"{key}: file={actual}, requested={requested}"
                    for key, (actual, requested) in conflicts.items()
                )
                raise ValueError(
                    f"Split metadata in {split_file} conflicts with this run: {detail}"
                )

            return _apply_patient_assignments(
                samples,
                assignments,
                seed=seed,
                fold=fold,
                n_splits=n_splits,
                stratify_mode=stratify_mode,
                source=str(split_file),
            )

    split = create_stratified_group_split(
        samples,
        n_splits=n_splits,
        fold=fold,
        seed=seed,
        stratify_mode=stratify_mode,
    )

    if split_file is not None:
        save_patient_split(split_file, samples, split)

    return split


def save_sample_manifest(
    path: Path,
    samples: Sequence[PatchSample],
    split: SplitIndices,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_set = {int(index) for index in split.train_indices}

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "path",
                "stain",
                "patient_id",
                "binary_class",
                "binary_label",
                "histologic_subtype",
                "stratum",
                "split",
            ]
        )
        for index, sample in enumerate(samples):
            writer.writerow(
                [
                    str(sample.path),
                    sample.stain,
                    sample.patient_id,
                    sample.class_name,
                    sample.label,
                    sample.subtype,
                    sample.binary_subtype_stratum,
                    "train" if index in train_set else "validation",
                ]
            )


def split_summary(
    samples: Sequence[PatchSample],
    split: SplitIndices,
) -> dict:
    result: dict[str, object] = {
        "source": split.source,
        "seed": split.seed,
        "fold": split.fold,
        "n_splits": split.n_splits,
        "stratify_mode": split.stratify_mode,
    }

    for name, indices in (
        ("train", split.train_indices),
        ("validation", split.val_indices),
    ):
        selected = [samples[int(index)] for index in indices]
        result[name] = {
            "patches": len(selected),
            "patients": len({sample.patient_id for sample in selected}),
            "binary_class_counts": dict(
                sorted(Counter(sample.class_name for sample in selected).items())
            ),
            "stratum_counts": dict(
                sorted(
                    Counter(
                        sample.binary_subtype_stratum for sample in selected
                    ).items()
                )
            ),
        }

    train_patients = {
        samples[int(index)].patient_id for index in split.train_indices
    }
    val_patients = {samples[int(index)].patient_id for index in split.val_indices}
    result["patient_overlap"] = sorted(train_patients & val_patients)
    return result


def save_split_summary(
    path: Path,
    samples: Sequence[PatchSample],
    split: SplitIndices,
) -> dict:
    summary = split_summary(samples, split)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
