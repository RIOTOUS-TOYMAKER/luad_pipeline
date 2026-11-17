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

DEFAULT_WSI_ROOT = Path("path/to/the/WSI_folder")
DEFAULT_GEOJSON_ROOT = Path(
    "path/to/the/geojson_folder"
)

####

SUPPORTED_EXPERIMENTS = tuple(
    experiment
    for stain in SUPPORTED_STAINS
    for experiment in (stain, f"{stain}_nr")
)
SUPPORTED_INPUT_TYPES = ("standard", "post-xenium")
SUPPORTED_GEOJSON_PROFILES = ("auto", "standard", "xenium")


@dataclass(frozen=True)
class WSIInferenceConfig:
    """Configuration shared by conventional and post-Xenium WSI inference."""

    experiment: str
    case: int
    checkpoint: Path
    input_type: str = "standard"
    wsi_root: Path = DEFAULT_WSI_ROOT
    geojson_root: Path = DEFAULT_GEOJSON_ROOT
    output_root: Path = Path(".")
    output_dir_override: Path | None = None
    results_csv_override: Path | None = None
    wsi_path_override: Path | None = None
    geojson_path_override: Path | None = None
    target_image_override: Path | None = None
    input_mpp_override: tuple[float, float] | None = None
    model_mpp: float = 0.2296
    architecture: str = "auto"
    tile_size: int = 512
    stride: int = 256
    batch_size: int = 128
    invasion_threshold: float = 0.5
    roi_overlap_threshold: float = 0.30
    saturation_threshold: int = 20
    min_tissue_ratio: float = 0.10
    min_area_mm2: float = 0.58
    invasion_class_index: int = 1
    device: str = "auto"
    amp: bool = False
    strict_checkpoint: bool = True
    validate_only: bool = False
    geojson_profile: str = "auto"
    classification_name: str = "AI invasion"
    classification_color: tuple[int, int, int] = (255, 0, 0)
    geojson_simplify_um: float = 1.0
    clip_output_to_roi: bool | None = None

    def __post_init__(self) -> None:
        if self.experiment not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Unsupported experiment {self.experiment!r}. "
                f"Choose from {SUPPORTED_EXPERIMENTS}."
            )
        if self.input_type not in SUPPORTED_INPUT_TYPES:
            raise ValueError(
                f"input_type must be one of {SUPPORTED_INPUT_TYPES}; "
                f"got {self.input_type!r}."
            )
        if self.input_type == "post-xenium" and self.stain != "HE":
            raise ValueError(
                "Post-Xenium input is an H&E image, so experiment must be "
                "'HE' or 'HE_nr'."
            )
        if self.input_type == "post-xenium":
            if self.wsi_path_override is None:
                raise ValueError(
                    "--wsi-path is required for --input-type post-xenium."
                )
            if self.geojson_path_override is None:
                raise ValueError(
                    "--geojson-path/--roi-geojson is required for "
                    "--input-type post-xenium."
                )
        if self.case < 1:
            raise ValueError("case must be 1 or greater.")
        canonical_architecture = normalize_architecture_name(
            self.architecture, allow_auto=True
        )
        object.__setattr__(self, "architecture", canonical_architecture)
        if self.tile_size < 1 or self.stride < 1 or self.batch_size < 1:
            raise ValueError("tile_size, stride, and batch_size must be positive.")
        if not self.model_mpp > 0:
            raise ValueError("model_mpp must be positive.")
        if self.input_mpp_override is not None:
            if len(self.input_mpp_override) != 2 or any(
                not value > 0 for value in self.input_mpp_override
            ):
                raise ValueError("input_mpp_override must contain two positive values.")
        for name, value in (
            ("invasion_threshold", self.invasion_threshold),
            ("roi_overlap_threshold", self.roi_overlap_threshold),
            ("min_tissue_ratio", self.min_tissue_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not 0 <= self.saturation_threshold <= 255:
            raise ValueError("saturation_threshold must be between 0 and 255.")
        if self.min_area_mm2 < 0:
            raise ValueError("min_area_mm2 must be non-negative.")
        if self.invasion_class_index < 0:
            raise ValueError("invasion_class_index must be non-negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.geojson_profile not in SUPPORTED_GEOJSON_PROFILES:
            raise ValueError(
                f"geojson_profile must be one of {SUPPORTED_GEOJSON_PROFILES}."
            )
        if len(self.classification_color) != 3 or any(
            channel < 0 or channel > 255
            for channel in self.classification_color
        ):
            raise ValueError(
                "classification_color must contain three integers from 0 to 255."
            )
        if self.geojson_simplify_um < 0:
            raise ValueError("geojson_simplify_um must be non-negative.")

    @property
    def use_reinhard(self) -> bool:
        return not self.experiment.endswith("_nr")

    @property
    def stain(self) -> str:
        return self.experiment.removesuffix("_nr")

    @property
    def case_stem(self) -> str:
        return f"p{self.case}_{self.stain}"

    @property
    def wsi_path(self) -> Path:
        if self.wsi_path_override is not None:
            return self.wsi_path_override
        return self.wsi_root / f"{self.case_stem}.ndpi"

    @property
    def geojson_path(self) -> Path:
        if self.geojson_path_override is not None:
            return self.geojson_path_override
        return self.geojson_root / self.stain / f"{self.case_stem}.ndpi.geojson"

    @property
    def target_image(self) -> Path | None:
        if not self.use_reinhard:
            return None
        return self.target_image_override or default_target_image(self.stain)

    @property
    def source_patch_size(self) -> int:
        return self.tile_size

    def resolved_model_input_size(self, resolved_architecture: str) -> int:
        return model_input_size_for_architecture(
            resolved_architecture,
            source_patch_size=self.source_patch_size,
        )

    @property
    def resolved_geojson_profile(self) -> str:
        if self.geojson_profile != "auto":
            return self.geojson_profile
        return "xenium" if self.input_type == "post-xenium" else "standard"

    @property
    def resolved_clip_output_to_roi(self) -> bool:
        if self.clip_output_to_roi is not None:
            return self.clip_output_to_roi
        return self.input_type == "post-xenium"

    @property
    def output_tag(self) -> str:
        return "post_xenium" if self.input_type == "post-xenium" else "standard"

    def output_dir_for(self, resolved_architecture: str) -> Path:
        if self.output_dir_override is not None:
            return self.output_dir_override
        architecture = str(normalize_architecture_name(resolved_architecture))
        if self.input_type == "post-xenium":
            name = (
                f"{self.case}_post_xenium_{self.experiment}_{architecture}"
                "_tumor_roi_invasion_result"
            )
        else:
            name = (
                f"{self.case}_{self.experiment}_{architecture}"
                "_tumor_roi_invasion_result"
            )
        return self.output_root / name

    def results_csv_for(self, resolved_architecture: str) -> Path:
        if self.results_csv_override is not None:
            return self.results_csv_override
        architecture = str(normalize_architecture_name(resolved_architecture))
        suffix = "_post_xenium" if self.input_type == "post-xenium" else ""
        return (
            self.output_root
            / f"{self.experiment}_{architecture}{suffix}_invasion_size_inference.csv"
        )

    @property
    def output_dir(self) -> Path:
        architecture = self.architecture if self.architecture != "auto" else "convnext_small"
        return self.output_dir_for(architecture)

    @property
    def results_csv(self) -> Path:
        architecture = self.architecture if self.architecture != "auto" else "convnext_small"
        return self.results_csv_for(architecture)

    @property
    def output_geojson_name(self) -> str:
        if self.resolved_geojson_profile == "xenium":
            return "invasion_regions_xenium.geojson"
        return "invasion_regions.geojson"

    @property
    def coordinate_space_description(self) -> str:
        if self.input_type == "post-xenium":
            return "original post-Xenium H&E level-0 pixels"
        return "input WSI level-0 pixels"

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        data.update(
            {
                "stain": self.stain,
                "use_reinhard": self.use_reinhard,
                "wsi_path": str(self.wsi_path),
                "geojson_path": str(self.geojson_path),
                "target_image": (
                    str(self.target_image) if self.target_image is not None else None
                ),
                "output_dir": str(self.output_dir),
                "results_csv": str(self.results_csv),
                "resolved_geojson_profile": self.resolved_geojson_profile,
                "resolved_clip_output_to_roi": self.resolved_clip_output_to_roi,
                "output_geojson_name": self.output_geojson_name,
                "coordinate_space": self.coordinate_space_description,
                "source_patch_size": self.source_patch_size,
                "model_input_size": (
                    self.resolved_model_input_size(self.architecture)
                    if self.architecture != "auto"
                    else None
                ),
                "spatial_preprocessing": (
                    spatial_preprocessing_spec(self.architecture)
                    if self.architecture != "auto"
                    else None
                ),
                "resize_or_center_crop": self.architecture == "vit_b_16",
                "input_normalization": {
                    "name": "ImageNet-1K",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }
        )
        return data
