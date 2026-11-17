from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from architectures import (
    model_input_size_for_architecture,
    normalize_architecture_name,
    spatial_preprocessing_spec,
)
from config import SUPPORTED_STAINS, default_target_image

####

DEFAULT_TEST_ROOT = Path("path/to/the/test/dataset")

####


SUPPORTED_EXPERIMENTS = tuple(
    experiment
    for stain in SUPPORTED_STAINS
    for experiment in (stain, f"{stain}_nr")
)
EXPECTED_CLASSES = ("in_situ", "invasion")


@dataclass(frozen=True)
class PatchEvaluationConfig:
    """Configuration for one independent patch-level performance test."""

    experiment: str
    checkpoint: Path
    seed: int = 1
    fold: int = 0
    test_root: Path = DEFAULT_TEST_ROOT
    test_dir_override: Path | None = None
    output_root: Path = Path(".")
    target_image_override: Path | None = None
    architecture: str = "auto"

    # Backwards-compatible field name: source patch size before any ViT crop.
    model_input_size: int = 512

    batch_size: int = 60
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    strict_checkpoint: bool = True
    positive_class: str = "invasion"
    plot_dpi: int = 800
    validate_only: bool = False

    integrated_gradients: bool = False
    ig_baseline: str = "mean"
    ig_target: str = "predicted"
    ig_n_steps: int = 4000
    ig_internal_batch_size: int = 20
    ig_overlay_alpha: float = 0.6
    ig_max_images: int | None = None
    ig_only_misclassified: bool = False
    pixel_size_um: float = 0.2296
    scale_bar_um: float = 50.0
    ig_dpi: int = 800

    def __post_init__(self) -> None:
        if self.experiment not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Unsupported experiment {self.experiment!r}. "
                f"Choose from {SUPPORTED_EXPERIMENTS}."
            )
        if self.seed < 0:
            raise ValueError("seed must be zero or greater.")
        if self.fold < 0:
            raise ValueError("fold must be zero or greater.")
        canonical_architecture = normalize_architecture_name(
            self.architecture, allow_auto=True
        )
        object.__setattr__(self, "architecture", canonical_architecture)
        if self.model_input_size < 1:
            raise ValueError("source patch size must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be zero or greater.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.positive_class not in EXPECTED_CLASSES:
            raise ValueError(
                f"positive_class must be one of {EXPECTED_CLASSES}; "
                f"got {self.positive_class!r}."
            )
        if self.plot_dpi < 1:
            raise ValueError("plot_dpi must be positive.")
        if self.ig_baseline not in {"zero", "mean"}:
            raise ValueError("ig_baseline must be 'zero' or 'mean'.")
        if self.ig_target not in {"predicted", "true"}:
            raise ValueError("ig_target must be 'predicted' or 'true'.")
        if self.ig_n_steps < 2:
            raise ValueError("ig_n_steps must be at least 2.")
        if self.ig_internal_batch_size < 1:
            raise ValueError("ig_internal_batch_size must be at least 1.")
        if not 0.0 <= self.ig_overlay_alpha <= 1.0:
            raise ValueError("ig_overlay_alpha must be between 0 and 1.")
        if self.ig_max_images is not None and self.ig_max_images < 1:
            raise ValueError("ig_max_images must be at least 1 when specified.")
        if self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive.")
        if self.scale_bar_um <= 0:
            raise ValueError("scale_bar_um must be positive.")
        if self.ig_dpi < 1:
            raise ValueError("ig_dpi must be positive.")

    @property
    def source_patch_size(self) -> int:
        return self.model_input_size

    def resolved_model_input_size(self, resolved_architecture: str) -> int:
        return model_input_size_for_architecture(
            resolved_architecture,
            source_patch_size=self.source_patch_size,
        )

    @property
    def use_reinhard(self) -> bool:
        return not self.experiment.endswith("_nr")

    @property
    def stain(self) -> str:
        return self.experiment.removesuffix("_nr")

    @property
    def test_dir(self) -> Path:
        return self.test_dir_override or self.test_root / self.stain

    @property
    def target_image(self) -> Path | None:
        if not self.use_reinhard:
            return None
        return self.target_image_override or default_target_image(self.stain)

    def output_dir_for(self, resolved_architecture: str) -> Path:
        architecture = str(normalize_architecture_name(resolved_architecture))
        return (
            self.output_root
            / "patch_test_results"
            / self.experiment
            / architecture
            / f"seed_{self.seed:02d}_fold_{self.fold}"
        )

    def results_csv_for(self, resolved_architecture: str) -> Path:
        architecture = str(normalize_architecture_name(resolved_architecture))
        return (
            self.output_root
            / "patch_test_results"
            / f"{self.experiment}_{architecture}_patch_test_results.csv"
        )

    @property
    def output_dir(self) -> Path:
        architecture = (
            self.architecture if self.architecture != "auto" else "convnext_small"
        )
        return self.output_dir_for(architecture)

    @property
    def results_csv(self) -> Path:
        architecture = (
            self.architecture if self.architecture != "auto" else "convnext_small"
        )
        return self.results_csv_for(architecture)

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        explicit_architecture = (
            self.architecture if self.architecture != "auto" else None
        )
        data.update(
            {
                "stain": self.stain,
                "use_reinhard": self.use_reinhard,
                "test_dir": str(self.test_dir),
                "target_image": (
                    str(self.target_image) if self.target_image is not None else None
                ),
                "output_dir": str(self.output_dir),
                "results_csv": str(self.results_csv),
                "expected_classes": list(EXPECTED_CLASSES),
                "source_patch_size": self.source_patch_size,
                "model_input_size": (
                    self.resolved_model_input_size(explicit_architecture)
                    if explicit_architecture is not None
                    else None
                ),
                "spatial_preprocessing": (
                    spatial_preprocessing_spec(explicit_architecture)
                    if explicit_architecture is not None
                    else None
                ),
                "resize_or_center_crop": explicit_architecture == "vit_b_16",
                "input_normalization": {
                    "name": "ImageNet-1K",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }
        )
        return data
