from __future__ import annotations

import argparse

from architectures import SUPPORTED_ARCHITECTURES, normalize_architecture_name
from pathlib import Path

from .runner import run_wsi_inference
from .settings import (
    DEFAULT_GEOJSON_ROOT,
    DEFAULT_WSI_ROOT,
    SUPPORTED_EXPERIMENTS,
    SUPPORTED_GEOJSON_PROFILES,
    SUPPORTED_INPUT_TYPES,
    WSIInferenceConfig,
)


def build_parser(
    *,
    default_experiment: str | None = None,
    default_input_type: str = "standard",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure WSI-level invasive regions from either a conventional WSI "
            "or a post-Xenium H&E OME-TIFF. Inference uses ImageNet-1K mean/std."
        )
    )
    parser.add_argument(
        "--input-type",
        choices=SUPPORTED_INPUT_TYPES,
        default=default_input_type,
        help=(
            "standard: retain the conventional level-0 512/256-pixel sampling; "
            "post-xenium: preserve the training-scale physical field of view and "
            "export Xenium-style annotations by default."
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=SUPPORTED_EXPERIMENTS,
        default=default_experiment,
        required=default_experiment is None,
    )
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        "--ckpt-path",
        dest="checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint file or training output directory. A directory is "
            "resolved to best_net_state_dict.pth or one best-*.ckpt."
        ),
    )
    parser.add_argument("--wsi-root", type=Path, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--geojson-root", type=Path, default=DEFAULT_GEOJSON_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory for this single run.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Override the cumulative case/invasion_size CSV path.",
    )
    parser.add_argument("--wsi-path", type=Path, default=None)
    parser.add_argument(
        "--geojson-path",
        "--roi-geojson",
        dest="geojson_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--target-image",
        "--target-img-path",
        dest="target_image",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--input-mpp",
        type=float,
        nargs="+",
        default=None,
        metavar="MPP",
        help=(
            "Override source-image MPP. Supply one value for isotropic pixels or "
            "two values for X and Y. Post-Xenium H&E may use --input-mpp 0.2738 "
            "when its OME metadata is missing or incorrect."
        ),
    )
    parser.add_argument(
        "--model-mpp",
        type=float,
        default=0.2296,
        help="Physical pixel size used when the patch classifier was trained.",
    )
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
        "--tile-size",
        type=int,
        default=512,
        help=(
            "Source patch/tile size at the training scale. ViT-B/16 then "
            "applies Resize(246) and CenterCrop(224) before the network."
        ),
    )
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--invasion-threshold", type=float, default=0.5)
    parser.add_argument("--roi-overlap-threshold", type=float, default=0.30)
    parser.add_argument("--saturation-threshold", type=int, default=20)
    parser.add_argument("--min-tissue-ratio", type=float, default=0.10)
    parser.add_argument("--min-area-mm2", type=float, default=0.58)
    parser.add_argument("--invasion-class-index", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA autocast for inference.",
    )
    parser.add_argument(
        "--strict-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every model key to match the checkpoint.",
    )
    parser.add_argument(
        "--geojson-profile",
        choices=SUPPORTED_GEOJSON_PROFILES,
        default="auto",
        help="auto selects standard output for conventional WSI and Xenium output for post-Xenium H&E.",
    )
    parser.add_argument("--classification-name", default="AI invasion")
    parser.add_argument(
        "--classification-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(255, 0, 0),
    )
    parser.add_argument(
        "--geojson-simplify-um",
        type=float,
        default=1.0,
        help="Topology-preserving boundary simplification in micrometres.",
    )
    parser.add_argument(
        "--clip-output-to-roi",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Clip output polygons to the input tumor ROI. Auto default: false "
            "for standard WSI, true for post-Xenium H&E."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs/model/ROI/MPP without performing tile inference.",
    )
    return parser


def _parse_input_mpp(
    parser: argparse.ArgumentParser,
    values: list[float] | None,
) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    if len(values) == 2:
        return float(values[0]), float(values[1])
    parser.error("--input-mpp accepts either one value or two values (X Y).")
    raise AssertionError("unreachable")


def main(
    *,
    default_experiment: str | None = None,
    default_input_type: str = "standard",
) -> None:
    parser = build_parser(
        default_experiment=default_experiment,
        default_input_type=default_input_type,
    )
    args = parser.parse_args()
    config = WSIInferenceConfig(
        experiment=args.experiment,
        case=args.case,
        checkpoint=args.checkpoint,
        input_type=args.input_type,
        wsi_root=args.wsi_root,
        geojson_root=args.geojson_root,
        output_root=args.output_root,
        output_dir_override=args.out_dir,
        results_csv_override=args.results_csv,
        wsi_path_override=args.wsi_path,
        geojson_path_override=args.geojson_path,
        target_image_override=args.target_image,
        input_mpp_override=_parse_input_mpp(parser, args.input_mpp),
        model_mpp=args.model_mpp,
        architecture=args.architecture,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        invasion_threshold=args.invasion_threshold,
        roi_overlap_threshold=args.roi_overlap_threshold,
        saturation_threshold=args.saturation_threshold,
        min_tissue_ratio=args.min_tissue_ratio,
        min_area_mm2=args.min_area_mm2,
        invasion_class_index=args.invasion_class_index,
        device=args.device,
        amp=args.amp,
        strict_checkpoint=args.strict_checkpoint,
        validate_only=args.validate_only,
        geojson_profile=args.geojson_profile,
        classification_name=args.classification_name,
        classification_color=tuple(args.classification_color),
        geojson_simplify_um=args.geojson_simplify_um,
        clip_output_to_roi=args.clip_output_to_roi,
    )
    run_wsi_inference(config)
