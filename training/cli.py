from __future__ import annotations

import argparse
from pathlib import Path

from architectures import SUPPORTED_ARCHITECTURES, normalize_architecture_name
from config import (
    DEFAULT_DATA_ROOT,
    ExperimentConfig,
    SUPPORTED_STAINS,
)
from .runner import run_experiment


def build_parser(
    *,
    default_stain: str | None = None,
    default_reinhard: bool | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the lung adenocarcinoma patch classifier with a "
            "patient-grouped StratifiedGroupKFold split."
        )
    )
    parser.add_argument(
        "--stain",
        choices=SUPPORTED_STAINS,
        default=default_stain,
        required=default_stain is None,
    )
    parser.add_argument(
        "--reinhard",
        action=argparse.BooleanOptionalAction,
        default=False if default_reinhard is None else default_reinhard,
        help="Enable or disable Reinhard normalization.",
    )
    parser.add_argument(
        "--architecture",
        type=lambda value: normalize_architecture_name(value),
        choices=SUPPORTED_ARCHITECTURES,
        default="convnext_small",
        help=(
            "Backbone architecture. Aliases such as efficientnet and "
            "visiontransformer are accepted and normalized to efficientnet_b6 "
            "and vit_b_16."
        ),
    )
    parser.add_argument(
        "--source-patch-size",
        "--model-input-size",
        dest="model_input_size",
        type=int,
        default=512,
        help=(
            "Source square patch size before architecture-specific preprocessing. "
            "All architectures keep this size except ViT-B/16, which applies "
            "Resize(246) then CenterCrop(224). --model-input-size is retained "
            "as a backwards-compatible alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--stratify-mode",
        choices=("binary", "binary_subtype"),
        default="binary_subtype",
        help=(
            "binary_subtype preserves the existing intent of balancing "
            "histologic subtypes within the two binary classes."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-image", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Patient-level CSV. If it exists, reuse it; if it does not exist, "
            "create it. Reuse the same file across stains for identical case "
            "assignments."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument(
        "--strict-filenames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every patch filename to begin <stain>_pNN_.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and audit the split without starting model training.",
    )
    return parser


def main(
    *,
    default_stain: str | None = None,
    default_reinhard: bool | None = None,
) -> None:
    parser = build_parser(
        default_stain=default_stain,
        default_reinhard=default_reinhard,
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        stain=args.stain,
        use_reinhard=args.reinhard,
        architecture=args.architecture,
        model_input_size=args.model_input_size,
        seed=args.seed,
        fold=args.fold,
        n_splits=args.n_splits,
        stratify_mode=args.stratify_mode,
        data_root=args.data_root,
        target_image=args.target_image,
        output_root=args.output_root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        strict_filenames=args.strict_filenames,
        dry_run=args.dry_run,
    )
    run_experiment(config)
