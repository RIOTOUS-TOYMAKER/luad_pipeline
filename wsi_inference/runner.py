from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from shapely.geometry import box

from architectures import spatial_preprocessing_spec
from modeling.checkpoints import validate_training_metadata
from preprocess import build_inference_transform

from .geojson_io import load_roi_geometry
from .measurement import postprocess_probability_map
from .model_io import load_inference_model
from .output import (
    append_legacy_result,
    make_invasion_geojson,
    save_array_outputs,
    write_json,
)
from .sampling import build_sampling_geometry
from .settings import WSIInferenceConfig
from .tiling import infer_probability_map


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _validate_paths(config: WSIInferenceConfig) -> None:
    required = {
        "WSI": config.wsi_path,
        "GeoJSON": config.geojson_path,
        "checkpoint": config.checkpoint,
    }
    if config.target_image is not None:
        required["Reinhard target image"] = config.target_image
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Required input(s) not found:\n" + "\n".join(missing))


def run_wsi_inference(config: WSIInferenceConfig) -> dict:
    torch.set_float32_matmul_precision("high")
    np.set_printoptions(suppress=True, precision=5)
    _validate_paths(config)

    device = resolve_device(config.device)
    print(f"Input type: {config.input_type}")
    print(f"Experiment: {config.experiment}")
    print(f"Case: {config.case}")
    print(f"Device: {device}")
    print(f"WSI: {config.wsi_path}")
    print(f"Tumor ROI: {config.geojson_path}")

    roi_geometry = load_roi_geometry(config.geojson_path)

    # Resolve architecture first so ViT receives its dedicated
    # Resize(246) -> CenterCrop(224) preprocessing and a 224-input model.
    model, architecture, checkpoint_path = load_inference_model(
        config.checkpoint,
        architecture=config.architecture,
        num_classes=2,
        strict=config.strict_checkpoint,
        source_patch_size=config.source_patch_size,
    )
    resolved_model_input_size = config.resolved_model_input_size(architecture)
    transform = build_inference_transform(
        use_reinhard=config.use_reinhard,
        target_image=config.target_image,
        architecture=architecture,
    )
    validate_training_metadata(
        checkpoint_path,
        expected_experiment=config.experiment,
        expected_architecture=architecture,
        expected_input_size=resolved_model_input_size,
        expected_source_patch_size=config.source_patch_size,
    )
    model.to(device)
    print(f"Architecture: {architecture}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Input normalization: ImageNet-1K mean/std")
    print(
        f"Source patch size: {config.source_patch_size} x "
        f"{config.source_patch_size}"
    )
    print(
        f"Network input size: {resolved_model_input_size} x "
        f"{resolved_model_input_size}"
    )
    if architecture == "vit_b_16":
        print("ViT spatial preprocessing: Resize(246) -> CenterCrop(224)")

    output_dir = config.output_dir_for(architecture)
    results_csv = config.results_csv_for(architecture)

    try:
        from tiatoolbox.wsicore.wsireader import WSIReader
    except ImportError as exc:
        raise ImportError(
            "WSI inference requires TIAToolbox. Install the WSI requirements "
            "listed in requirements.txt."
        ) from exc

    wsi = WSIReader.open(str(config.wsi_path))
    sampling = build_sampling_geometry(wsi, config)
    slide_box = box(
        0,
        0,
        sampling.slide_width_px,
        sampling.slide_height_px,
    )
    if not roi_geometry.intersects(slide_box):
        raise ValueError(
            "The tumor ROI does not overlap the input image. Confirm that the "
            "GeoJSON and WSI use the same level-0 pixel coordinate system."
        )

    print(
        "Input MPP: "
        f"x={sampling.input_mpp_x:.6f}, "
        f"y={sampling.input_mpp_y:.6f} ({sampling.mpp_source})"
    )
    print(f"ROI bounds: {tuple(float(v) for v in roi_geometry.bounds)}")
    print(
        "Source tile footprint: "
        f"{sampling.source_tile_width_px} x "
        f"{sampling.source_tile_height_px} level-0 px"
    )
    print(
        "Source grid stride: "
        f"x={sampling.source_stride_x_px:.3f}, "
        f"y={sampling.source_stride_y_px:.3f} level-0 px"
    )
    if config.input_type == "post-xenium":
        print(
            "Post-Xenium scale adaptation: source tiles are resized to "
            f"{config.tile_size} x {config.tile_size} source-patch pixels "
            "before architecture-specific preprocessing."
        )
        if (
            sampling.input_mpp_x > config.model_mpp
            or sampling.input_mpp_y > config.model_mpp
        ):
            print(
                "Note: the post-Xenium image is coarser than the model scale; "
                "tiles are upsampled, but upsampling cannot restore detail that "
                "is absent from the source image."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = config.to_serializable_dict()
    run_config["resolved_architecture"] = architecture
    run_config["resolved_checkpoint"] = str(checkpoint_path)
    run_config["source_patch_size"] = config.source_patch_size
    run_config["model_input_size"] = resolved_model_input_size
    run_config["spatial_preprocessing"] = spatial_preprocessing_spec(architecture)
    run_config["resize_or_center_crop"] = architecture == "vit_b_16"
    run_config["sampling_geometry"] = sampling.to_serializable_dict()
    run_config["output_dir"] = str(output_dir)
    run_config["results_csv"] = str(results_csv)
    write_json(output_dir / "run_config.json", run_config)

    if config.validate_only:
        print("Validation completed; tile inference was not run.")
        return run_config

    tile_result = infer_probability_map(
        wsi=wsi,
        roi_geometry=roi_geometry,
        sampling_geometry=sampling,
        model=model,
        transform=transform,
        device=device,
        batch_size=config.batch_size,
        roi_overlap_threshold=config.roi_overlap_threshold,
        saturation_threshold=config.saturation_threshold,
        min_tissue_ratio=config.min_tissue_ratio,
        invasion_class_index=config.invasion_class_index,
        amp=config.amp,
    )
    print(
        "Tiles: "
        f"grid={tile_result.grid_tiles}, "
        f"ROI-qualified={tile_result.roi_tiles}, "
        f"inferred={tile_result.tissue_tiles}"
    )

    measurement = postprocess_probability_map(
        tile_result.probability_map,
        xs=tile_result.xs,
        ys=tile_result.ys,
        sampling_geometry=sampling,
        invasion_threshold=config.invasion_threshold,
        min_area_mm2=config.min_area_mm2,
    )

    save_array_outputs(output_dir, tile_result, measurement)
    measurement.components.to_csv(
        output_dir / "invasion_components.csv",
        index=False,
    )
    geojson_path = output_dir / config.output_geojson_name
    write_json(
        geojson_path,
        make_invasion_geojson(
            measurement,
            tile_result=tile_result,
            sampling_geometry=sampling,
            roi_geometry=roi_geometry,
            config=config,
        ),
    )

    summary = {
        "case": config.case,
        "input_type": config.input_type,
        "experiment": config.experiment,
        "stain": config.stain,
        "reinhard": config.use_reinhard,
        "architecture": architecture,
        "source_patch_size": config.source_patch_size,
        "model_input_size": resolved_model_input_size,
        "checkpoint": str(checkpoint_path),
        "wsi_path": str(config.wsi_path),
        "geojson_path": str(config.geojson_path),
        "output_geojson": str(geojson_path),
        "coordinate_space": config.coordinate_space_description,
        "input_mpp_x": sampling.input_mpp_x,
        "input_mpp_y": sampling.input_mpp_y,
        "source_tile_width_px": sampling.source_tile_width_px,
        "source_tile_height_px": sampling.source_tile_height_px,
        "source_stride_x_px": sampling.source_stride_x_px,
        "source_stride_y_px": sampling.source_stride_y_px,
        "grid_tiles": tile_result.grid_tiles,
        "roi_tiles": tile_result.roi_tiles,
        "inferred_tiles": tile_result.tissue_tiles,
        "connected_components": int(len(measurement.components)),
        "invasion_size_mm": measurement.invasion_size_mm,
        "largest_component": (
            measurement.components.iloc[0].to_dict()
            if not measurement.components.empty
            else None
        ),
        "grid_step_x_um": measurement.grid_step_x_um,
        "grid_step_y_um": measurement.grid_step_y_um,
        "pixel_area_mm2": measurement.pixel_area_mm2,
        "min_area_px": measurement.min_area_px,
    }
    write_json(output_dir / "inference_summary.json", summary)
    append_legacy_result(
        results_csv,
        case=config.case,
        invasion_size_mm=measurement.invasion_size_mm,
    )

    if measurement.components.empty:
        print("No invasion")
    else:
        largest = measurement.components.iloc[0]
        print(f"Farthest distance (mm): {largest['farthest_mm']:.6f}")
        print(f"Major axis length (mm): {largest['major_axis_mm']:.6f}")
    print(f"Output directory: {output_dir}")
    print(f"GeoJSON: {geojson_path}")
    print(f"Results CSV: {results_csv}")
    return summary
