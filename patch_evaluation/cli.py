from __future__ import annotations

import argparse

from architectures import SUPPORTED_ARCHITECTURES, normalize_architecture_name
from pathlib import Path

from .runner import run_patch_evaluation
from .settings import (
    DEFAULT_TEST_ROOT,
    SUPPORTED_EXPERIMENTS,
    PatchEvaluationConfig,
)


def build_parser(
    *,
    default_experiment: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an independent patch test set with ImageNet-1K "
            "normalization. Integrated Gradients is disabled unless explicitly "
            "requested."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=SUPPORTED_EXPERIMENTS,
        default=default_experiment,
        required=default_experiment is None,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint file or training output directory. A directory resolves "
            "to best_net_state_dict.pth or one best-*.ckpt."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--target-image", type=Path, default=None)
    parser.add_argument(
        "--architecture",
        type=lambda value: normalize_architecture_name(value, allow_auto=True),
        choices=("auto", *SUPPORTED_ARCHITECTURES),
        default="auto",
        help=(
            "Infer from checkpoint keys by default. Aliases efficientnet and "
            "visiontransformer are accepted."
        ),
    )
    parser.add_argument(
        "--source-patch-size",
        "--model-input-size",
        dest="model_input_size",
        type=int,
        default=512,
        help=(
            "Source square patch size. ViT-B/16 alone applies Resize(246) "
            "followed by CenterCrop(224); other architectures retain this size. "
            "--model-input-size is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA autocast for ordinary prediction inference only.",
    )
    parser.add_argument(
        "--strict-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plot-dpi", type=int, default=800)
    parser.add_argument(
        "--positive-class",
        choices=("in_situ", "invasion"),
        default="invasion",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the test set, preprocessing, and checkpoint without inference.",
    )

    ig = parser.add_argument_group("Integrated Gradients (disabled by default)")
    ig.add_argument(
        "--integrated-gradients",
        "--ig",
        action="store_true",
        default=False,
        help="Run Captum Integrated Gradients after the performance test.",
    )
    ig.add_argument(
        "--ig-baseline",
        choices=("mean", "zero"),
        default="mean",
    )
    ig.add_argument(
        "--ig-target",
        choices=("predicted", "true"),
        default="predicted",
    )
    ig.add_argument("--ig-n-steps", type=int, default=4000)
    ig.add_argument("--ig-internal-batch-size", type=int, default=20)
    ig.add_argument("--ig-overlay-alpha", type=float, default=0.6)
    ig.add_argument(
        "--ig-max-images",
        type=int,
        default=None,
        help="Limit IG to the first N selected patches; default is all patches.",
    )
    ig.add_argument(
        "--ig-only-misclassified",
        action="store_true",
        help="Generate IG only for incorrectly classified patches.",
    )
    ig.add_argument("--pixel-size-um", type=float, default=0.2296)
    ig.add_argument("--scale-bar-um", type=float, default=50.0)
    ig.add_argument("--ig-dpi", type=int, default=800)
    return parser


def main(*, default_experiment: str | None = None) -> None:
    args = build_parser(default_experiment=default_experiment).parse_args()
    config = PatchEvaluationConfig(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        seed=args.seed,
        fold=args.fold,
        test_root=args.test_root,
        test_dir_override=args.test_dir,
        output_root=args.output_root,
        target_image_override=args.target_image,
        architecture=args.architecture,
        model_input_size=args.model_input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        strict_checkpoint=args.strict_checkpoint,
        positive_class=args.positive_class,
        plot_dpi=args.plot_dpi,
        validate_only=args.validate_only,
        integrated_gradients=args.integrated_gradients,
        ig_baseline=args.ig_baseline,
        ig_target=args.ig_target,
        ig_n_steps=args.ig_n_steps,
        ig_internal_batch_size=args.ig_internal_batch_size,
        ig_overlay_alpha=args.ig_overlay_alpha,
        ig_max_images=args.ig_max_images,
        ig_only_misclassified=args.ig_only_misclassified,
        pixel_size_um=args.pixel_size_um,
        scale_bar_um=args.scale_bar_um,
        ig_dpi=args.ig_dpi,
    )
    run_patch_evaluation(config)
