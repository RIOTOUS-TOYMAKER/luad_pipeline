from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from architectures import (
    model_input_size_for_architecture,
    normalize_architecture_name,
)
from preprocess import IMAGENET1K_MEAN, IMAGENET1K_STD
from training.model import build_image_classifier


KNOWN_PREFIXES = ("model.net.", "module.net.", "net.", "module.", "model.")


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda while failing clearly for unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)


def resolve_checkpoint_path(path: Path) -> Path:
    """Accept either a checkpoint file or one training-output directory."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint file/directory not found: {path}")

    preferred = path / "best_net_state_dict.pth"
    if preferred.is_file():
        return preferred.resolve()

    candidates = sorted(path.glob("best-*.ckpt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"No best_net_state_dict.pth or best-*.ckpt was found in {path}."
        )
    raise RuntimeError(
        f"Multiple best-*.ckpt files were found in {path}; pass one file explicitly."
    )


def _torch_load(path: Path) -> Any:
    # These checkpoints are trusted project artifacts and Lightning checkpoints
    # may contain non-tensor metadata. The fallback supports older PyTorch.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]

    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping.")

    state_dict = {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("No tensors were found in the checkpoint state_dict.")
    return state_dict


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Choose the Lightning/DataParallel prefix removal with the best match."""
    model_keys = set(model.state_dict())
    candidates = [
        _strip_prefix(state_dict, prefix)
        for prefix in ("", *KNOWN_PREFIXES)
    ]
    return max(candidates, key=lambda item: len(model_keys & set(item)))


def infer_architecture(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = set(state_dict)
    unprefixed = {
        key.removeprefix("model.net.")
        .removeprefix("module.net.")
        .removeprefix("net.")
        .removeprefix("module.")
        .removeprefix("model.")
        for key in keys
    }

    if "encoder.pos_embedding" in unprefixed and "class_token" in unprefixed:
        return "vit_b_16"
    if any(key.startswith("layer3.35.") for key in unprefixed) and any(
        key.startswith("fc.") for key in unprefixed
    ):
        return "resnet152"
    if any(".attn.qkv." in key for key in unprefixed) or any(
        key.startswith("head.") for key in unprefixed
    ):
        return "swin_s"
    if any(".layer_scale" in key for key in unprefixed) or any(
        key.startswith("classifier.2.") for key in unprefixed
    ):
        return "convnext_small"
    if any(".fc1." in key for key in unprefixed) and any(
        key.startswith("features.8.") for key in unprefixed
    ):
        return "efficientnet_b6"

    raise ValueError(
        "Could not infer the checkpoint architecture. Supported automatic "
        "detection: ConvNeXt-Small, Swin-S, ResNet152, EfficientNet-B6, "
        "and ViT-B/16. Specify --architecture explicitly."
    )


def load_inference_model(
    checkpoint: Path,
    *,
    architecture: str = "auto",
    num_classes: int = 2,
    strict: bool = True,
    source_patch_size: int = 512,
    input_size: int | None = None,
) -> tuple[nn.Module, str, Path]:
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    raw_state_dict = extract_state_dict(_torch_load(checkpoint_path))
    resolved_architecture = (
        infer_architecture(raw_state_dict)
        if architecture == "auto"
        else str(normalize_architecture_name(architecture))
    )

    # No pretrained download is needed at inference because the checkpoint is
    # expected to contain the complete model state_dict. ViT resolves to 224
    # after Resize(246) -> CenterCrop(224); other models retain source size.
    resolved_input_size = (
        int(input_size)
        if input_size is not None
        else model_input_size_for_architecture(
            resolved_architecture,
            source_patch_size=source_patch_size,
        )
    )
    model = build_image_classifier(
        resolved_architecture,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
        input_size=resolved_input_size,
    )
    state_dict = normalize_state_dict_keys(raw_state_dict, model)

    try:
        incompatibility = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint/model mismatch for architecture {resolved_architecture!r}: "
            f"{checkpoint_path}\n{exc}"
        ) from exc

    if not strict and (incompatibility.missing_keys or incompatibility.unexpected_keys):
        print(f"Missing checkpoint keys: {incompatibility.missing_keys}")
        print(f"Unexpected checkpoint keys: {incompatibility.unexpected_keys}")

    model.eval()
    return model, resolved_architecture, checkpoint_path


def _normalization_matches_imagenet(metadata: Mapping[str, Any]) -> bool | None:
    """Return True/False when normalization is declared, otherwise None."""
    normalization = metadata.get("input_normalization")
    if not isinstance(normalization, Mapping):
        return None

    mean = normalization.get("mean")
    std = normalization.get("std")
    if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
        if len(mean) != 3 or len(std) != 3:
            return False
        return all(
            abs(float(observed) - expected) < 1e-8
            for observed, expected in zip(mean, IMAGENET1K_MEAN)
        ) and all(
            abs(float(observed) - expected) < 1e-8
            for observed, expected in zip(std, IMAGENET1K_STD)
        )

    name = str(normalization.get("name", "")).strip().lower()
    if name:
        return name in {"imagenet-1k", "imagenet1k", "imagenet 1k"}
    return None


def read_training_metadata(checkpoint_path: Path) -> tuple[dict[str, Any] | None, Path]:
    """Read run_config.json next to a resolved checkpoint when available."""
    run_config_path = Path(checkpoint_path).parent / "run_config.json"
    if not run_config_path.is_file():
        return None, run_config_path

    try:
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read training metadata: {run_config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Training metadata is not a JSON object: {run_config_path}")
    return payload, run_config_path


def validate_training_metadata(
    checkpoint_path: Path,
    *,
    expected_experiment: str | None = None,
    expected_seed: int | None = None,
    expected_fold: int | None = None,
    expected_architecture: str | None = None,
    expected_input_size: int | None = None,
    expected_source_patch_size: int | None = None,
) -> dict[str, Any] | None:
    """
    Validate experiment identity and ImageNet normalization when recorded.

    Missing metadata produces a warning because legacy checkpoints may not have
    run_config.json. An explicitly incompatible normalization raises an error so
    it cannot silently be evaluated with the ImageNet preprocessing pipeline.
    """
    metadata, run_config_path = read_training_metadata(checkpoint_path)
    if metadata is None:
        print(
            "Warning: no adjacent run_config.json was found, so the checkpoint's "
            "training-time normalization cannot be verified. Evaluation will use "
            "ImageNet-1K mean/std."
        )
        return None

    recorded_experiment = metadata.get("experiment_name") or metadata.get("experiment")
    if (
        expected_experiment is not None
        and recorded_experiment is not None
        and str(recorded_experiment) != expected_experiment
    ):
        raise ValueError(
            "Checkpoint experiment mismatch: "
            f"requested {expected_experiment!r}, metadata records "
            f"{recorded_experiment!r} in {run_config_path}."
        )

    for field, expected in (("seed", expected_seed), ("fold", expected_fold)):
        recorded = metadata.get(field)
        if expected is not None and recorded is not None and int(recorded) != expected:
            raise ValueError(
                f"Checkpoint {field} mismatch: requested {expected}, metadata "
                f"records {recorded} in {run_config_path}."
            )

    recorded_architecture = metadata.get("architecture")
    if expected_architecture is not None and recorded_architecture is not None:
        expected_canonical = normalize_architecture_name(expected_architecture)
        recorded_canonical = normalize_architecture_name(str(recorded_architecture))
        if recorded_canonical != expected_canonical:
            raise ValueError(
                "Checkpoint architecture mismatch: "
                f"requested {expected_canonical!r}, metadata records "
                f"{recorded_canonical!r} in {run_config_path}."
            )

    recorded_input_size = metadata.get("model_input_size")
    if (
        expected_input_size is not None
        and recorded_input_size is not None
        and int(recorded_input_size) != expected_input_size
    ):
        raise ValueError(
            "Checkpoint model_input_size mismatch: "
            f"requested {expected_input_size}, metadata records "
            f"{recorded_input_size} in {run_config_path}."
        )

    recorded_source_patch_size = metadata.get("source_patch_size")
    if (
        expected_source_patch_size is not None
        and recorded_source_patch_size is not None
        and int(recorded_source_patch_size) != expected_source_patch_size
    ):
        raise ValueError(
            "Checkpoint source_patch_size mismatch: "
            f"requested {expected_source_patch_size}, metadata records "
            f"{recorded_source_patch_size} in {run_config_path}."
        )

    matches = _normalization_matches_imagenet(metadata)
    if matches is False:
        raise ValueError(
            "Checkpoint metadata records preprocessing incompatible with the "
            f"ImageNet-1K evaluation transform: {run_config_path}"
        )
    if matches is None:
        print(
            "Warning: run_config.json does not record enough information to "
            "verify training-time normalization. Evaluation will use "
            "ImageNet-1K mean/std."
        )
    return metadata
