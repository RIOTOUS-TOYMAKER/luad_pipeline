from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from architectures import (
    SUPPORTED_ARCHITECTURES,
    classification_head_spec,
    model_input_size_for_architecture,
    normalize_architecture_name,
    pretrained_weight_name,
    spatial_preprocessing_spec,
)

####

DEFAULT_DATA_ROOT = Path("path/to/the/training_validation_dataset")
DEFAULT_TARGET_ROOT = Path("path/to/the/target_image_folder")

DEFAULT_TARGET_IMAGES: dict[str, Path] = {
    "HE": DEFAULT_TARGET_ROOT / "filename_of_the_representative_H&E_image.png",
    "EVG": DEFAULT_TARGET_ROOT / "filename_of_the_representative_EVG_image.png",
    "CK7+EVG": DEFAULT_TARGET_ROOT / "filename_of_the_representative_CK7+EVG_image.png",
    "CK7+VB": DEFAULT_TARGET_ROOT / "filename_of_the_representative_CK7+VB_image.png",
}

####

SUPPORTED_STAINS = tuple(DEFAULT_TARGET_IMAGES)


def default_target_image(stain: str) -> Path:
    preferred = DEFAULT_TARGET_IMAGES[stain]
    if preferred.is_file():
        return preferred
    return preferred


@dataclass(frozen=True)
class ExperimentConfig:
    stain: str
    use_reinhard: bool
    architecture: str = "convnext_small"

    # Kept under the old field name for command/API compatibility. It denotes
    # the source patch size before architecture-specific preprocessing.
    model_input_size: int = 512

    seed: int = 1
    fold: int = 0
    n_splits: int = 5
    stratify_mode: str = "binary_subtype"
    data_root: Path = DEFAULT_DATA_ROOT
    target_image: Path | None = None
    output_root: Path = Path(".")
    split_file: Path | None = None
    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 1e-3
    max_epochs: int = 400
    patience: int = 5
    min_delta: float = 0.001
    strict_filenames: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        canonical_architecture = normalize_architecture_name(self.architecture)
        object.__setattr__(self, "architecture", canonical_architecture)
        if self.stain not in SUPPORTED_STAINS:
            raise ValueError(
                f"Unsupported stain {self.stain!r}. Choose from {SUPPORTED_STAINS}."
            )
        if self.model_input_size < 1:
            raise ValueError("source patch size must be positive.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0 <= self.fold < self.n_splits:
            raise ValueError(
                f"fold must be between 0 and {self.n_splits - 1}; got {self.fold}."
            )
        if self.stratify_mode not in {"binary", "binary_subtype"}:
            raise ValueError(
                "stratify_mode must be 'binary' or 'binary_subtype'."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be zero or greater.")

    @property
    def source_patch_size(self) -> int:
        return self.model_input_size

    @property
    def resolved_model_input_size(self) -> int:
        return model_input_size_for_architecture(
            self.architecture,
            source_patch_size=self.source_patch_size,
        )

    @property
    def spatial_preprocessing(self) -> dict[str, object]:
        return spatial_preprocessing_spec(self.architecture)

    @property
    def experiment_name(self) -> str:
        return self.stain if self.use_reinhard else f"{self.stain}_nr"

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.stain

    @property
    def resolved_target_image(self) -> Path | None:
        if not self.use_reinhard:
            return None
        return self.target_image or default_target_image(self.stain)

    @property
    def output_dir(self) -> Path:
        return (
            self.output_root
            / "outputs"
            / self.experiment_name
            / self.architecture
            / f"seed_{self.seed:02d}_fold_{self.fold}"
        )

    @property
    def log_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def results_csv(self) -> Path:
        return (
            self.output_root
            / f"{self.experiment_name}_{self.architecture}_sgkf_training_results.csv"
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)

        # Override the legacy field with the actual network input size, and
        # record the original extraction size separately.
        result["source_patch_size"] = self.source_patch_size
        result["model_input_size"] = self.resolved_model_input_size
        result["data_dir"] = str(self.data_dir)
        result["resolved_target_image"] = (
            str(self.resolved_target_image)
            if self.resolved_target_image is not None
            else None
        )
        result["experiment_name"] = self.experiment_name
        result["architecture"] = self.architecture
        result["supported_architectures"] = list(SUPPORTED_ARCHITECTURES)
        result["pretrained_weights"] = pretrained_weight_name(self.architecture)
        result["classification_head"] = classification_head_spec()
        result["spatial_preprocessing"] = self.spatial_preprocessing
        result["resize_or_center_crop"] = self.architecture == "vit_b_16"
        result["output_dir"] = str(self.output_dir)
        result["results_csv"] = str(self.results_csv)
        result["input_normalization"] = {
            "name": "ImageNet-1K",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        return result
