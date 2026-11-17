"""Backward-compatible re-exports for shared checkpoint utilities."""

from modeling.checkpoints import (
    KNOWN_PREFIXES,
    extract_state_dict,
    infer_architecture,
    load_inference_model,
    normalize_state_dict_keys,
    read_training_metadata,
    resolve_checkpoint_path,
    resolve_device,
    validate_training_metadata,
)

__all__ = [
    "KNOWN_PREFIXES",
    "extract_state_dict",
    "infer_architecture",
    "load_inference_model",
    "normalize_state_dict_keys",
    "read_training_metadata",
    "resolve_checkpoint_path",
    "resolve_device",
    "validate_training_metadata",
]
