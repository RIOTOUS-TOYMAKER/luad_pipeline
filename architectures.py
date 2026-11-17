from __future__ import annotations

from typing import Literal, cast

Architecture = Literal[
    "convnext_small",
    "swin_s",
    "resnet152",
    "efficientnet_b6",
    "vit_b_16",
]

SUPPORTED_ARCHITECTURES: tuple[Architecture, ...] = (
    "convnext_small",
    "swin_s",
    "resnet152",
    "efficientnet_b6",
    "vit_b_16",
)

ARCHITECTURE_ALIASES: dict[str, Architecture] = {
    "convnext": "convnext_small",
    "swin": "swin_s",
    "resnet": "resnet152",
    "efficientnet": "efficientnet_b6",
    "visiontransformer": "vit_b_16",
    "vision_transformer": "vit_b_16",
    "vit": "vit_b_16",
}

ARCHITECTURE_DISPLAY_NAMES: dict[Architecture, str] = {
    "convnext_small": "ConvNeXt-Small",
    "swin_s": "Swin Transformer-S",
    "resnet152": "ResNet152",
    "efficientnet_b6": "EfficientNet-B6",
    "vit_b_16": "Vision Transformer-B/16",
}

PRETRAINED_WEIGHT_NAMES: dict[Architecture, str] = {
    "convnext_small": "ConvNeXt_Small_Weights.IMAGENET1K_V1",
    "swin_s": "Swin_S_Weights.IMAGENET1K_V1",
    "resnet152": "ResNet152_Weights.IMAGENET1K_V2",
    "efficientnet_b6": "EfficientNet_B6_Weights.IMAGENET1K_V1",
    "vit_b_16": "ViT_B_16_Weights.IMAGENET1K_V1",
}


def normalize_architecture_name(
    value: str,
    *,
    allow_auto: bool = False,
) -> Architecture | Literal["auto"]:
    normalized = str(value).strip().lower().replace("-", "_")
    if allow_auto and normalized == "auto":
        return "auto"
    normalized = ARCHITECTURE_ALIASES.get(normalized, normalized)  # type: ignore[assignment]
    if normalized not in SUPPORTED_ARCHITECTURES:
        allowed = ("auto", *SUPPORTED_ARCHITECTURES) if allow_auto else SUPPORTED_ARCHITECTURES
        raise ValueError(
            f"Unsupported architecture {value!r}; choose from {allowed}."
        )
    return cast(Architecture, normalized)


def architecture_display_name(value: str) -> str:
    architecture = normalize_architecture_name(value)
    assert architecture != "auto"
    return ARCHITECTURE_DISPLAY_NAMES[architecture]


def pretrained_weight_name(value: str) -> str:
    architecture = normalize_architecture_name(value)
    assert architecture != "auto"
    return PRETRAINED_WEIGHT_NAMES[architecture]


def classification_head_spec() -> list[dict[str, object]]:
    return [
        {"layer": "Linear", "out_features": 512},
        {"layer": "ReLU"},
        {"layer": "Dropout", "p": 0.5},
        {"layer": "Linear", "out_features": 128},
        {"layer": "ReLU"},
        {"layer": "Dropout", "p": 0.3},
        {"layer": "Linear", "out_features": "num_classes"},
    ]


VIT_RESIZE_SIZE = 246
VIT_CENTER_CROP_SIZE = 224


def model_input_size_for_architecture(
    value: str,
    *,
    source_patch_size: int = 512,
) -> int:
    """Return the square tensor size presented to the network.

    All CNN/Swin backbones receive the source patch unchanged. ViT-B/16 uses
    the historical project preprocessing: Resize(246) then CenterCrop(224).
    """
    if source_patch_size < 1:
        raise ValueError("source_patch_size must be positive.")
    architecture = normalize_architecture_name(value)
    assert architecture != "auto"
    return VIT_CENTER_CROP_SIZE if architecture == "vit_b_16" else source_patch_size


def spatial_preprocessing_spec(value: str) -> dict[str, object]:
    """Describe deterministic architecture-specific spatial preprocessing."""
    architecture = normalize_architecture_name(value)
    assert architecture != "auto"
    if architecture == "vit_b_16":
        return {
            "resize": VIT_RESIZE_SIZE,
            "center_crop": VIT_CENTER_CROP_SIZE,
            "interpolation": "bilinear",
            "antialias": True,
        }
    return {
        "resize": None,
        "center_crop": None,
        "interpolation": None,
        "antialias": None,
    }


def processed_pixel_size_um(
    value: str,
    *,
    source_patch_size: int,
    source_pixel_size_um: float,
) -> float:
    """Pixel size after architecture-specific resizing (for IG scale bars)."""
    if source_pixel_size_um <= 0:
        raise ValueError("source_pixel_size_um must be positive.")
    architecture = normalize_architecture_name(value)
    assert architecture != "auto"
    if architecture == "vit_b_16":
        return float(source_pixel_size_um) * source_patch_size / VIT_RESIZE_SIZE
    return float(source_pixel_size_um)
