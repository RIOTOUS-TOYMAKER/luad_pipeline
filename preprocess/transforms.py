from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from architectures import (
    VIT_CENTER_CROP_SIZE,
    VIT_RESIZE_SIZE,
    normalize_architecture_name,
)


IMAGENET1K_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET1K_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ReinhardNormalizeToTensor:
    """Apply torchstain Reinhard normalization and return CHW float32 [0, 1]."""

    def __init__(self, target_image: Path) -> None:
        try:
            import torchstain
        except ImportError as exc:
            raise ImportError(
                "Reinhard normalization requires torchstain. "
                "Install it with: pip install torchstain"
            ) from exc

        target_image = Path(target_image)
        if not target_image.is_file():
            raise FileNotFoundError(
                f"Reinhard target image not found: {target_image}"
            )

        self.to_tensor = transforms.ToTensor()
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(
            backend="torch"
        )

        with Image.open(target_image) as image_file:
            target = image_file.convert("RGB")
            target_tensor = self.to_tensor(target) * 255.0

        self.normalizer.fit(target_tensor)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        source = self.to_tensor(image.convert("RGB")) * 255.0
        normalized = self.normalizer.normalize(I=source)

        if not isinstance(normalized, torch.Tensor) or normalized.ndim != 3:
            raise RuntimeError(
                "torchstain returned an unexpected Reinhard output shape/type."
            )

        if normalized.shape[0] == 3:
            chw = normalized
        elif normalized.shape[-1] == 3:
            chw = normalized.permute(2, 0, 1)
        else:
            raise RuntimeError(
                f"Cannot identify the RGB channel dimension: {normalized.shape}"
            )

        return chw.contiguous().float() / 255.0


def imagenet1k_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD)


def denormalize_imagenet1k(
    tensor: torch.Tensor,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        view_shape = (3, 1, 1)
    elif tensor.ndim == 4 and tensor.shape[1] == 3:
        view_shape = (1, 3, 1, 1)
    else:
        raise ValueError(
            "Expected an RGB tensor with shape (3, H, W) or (B, 3, H, W); "
            f"got {tuple(tensor.shape)}."
        )

    mean = tensor.new_tensor(IMAGENET1K_MEAN).view(view_shape)
    std = tensor.new_tensor(IMAGENET1K_STD).view(view_shape)
    restored = tensor * std + mean
    return restored.clamp(0.0, 1.0) if clamp else restored


denormalize_imagenet = denormalize_imagenet1k


def _spatial_steps(architecture: str) -> list[object]:
    """Return architecture-specific PIL spatial transforms."""
    canonical = normalize_architecture_name(architecture)
    assert canonical != "auto"
    if canonical == "vit_b_16":
        return [
            transforms.Resize(
                VIT_RESIZE_SIZE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(VIT_CENTER_CROP_SIZE),
        ]
    return []


def build_transforms(
    *,
    use_reinhard: bool,
    target_image: Path | None,
    architecture: str = "convnext_small",
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and validation transforms.

    ViT-B/16 alone uses Resize(246) followed by CenterCrop(224), matching the
    historical ViT experiments. Every other architecture keeps the source
    patch size unchanged. ImageNet-1K channel normalization is shared by all.
    """
    canonical = normalize_architecture_name(architecture)
    assert canonical != "auto"

    if use_reinhard:
        if target_image is None:
            raise ValueError(
                "target_image is required when Reinhard normalization is enabled."
            )
        image_to_tensor = ReinhardNormalizeToTensor(target_image)
    else:
        image_to_tensor = transforms.ToTensor()

    normalize = imagenet1k_normalize()

    val_transform = transforms.Compose(
        [
            *_spatial_steps(canonical),
            image_to_tensor,
            normalize,
        ]
    )

    train_transform = transforms.Compose(
        [
            *_spatial_steps(canonical),
            transforms.RandomHorizontalFlip(p=0.5),
            image_to_tensor,
            normalize,
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
                inplace=False,
            ),
        ]
    )

    return train_transform, val_transform


def build_inference_transform(
    *,
    use_reinhard: bool,
    target_image: Path | None,
    architecture: str = "convnext_small",
) -> transforms.Compose:
    _, inference_transform = build_transforms(
        use_reinhard=use_reinhard,
        target_image=target_image,
        architecture=architecture,
    )
    return inference_transform
