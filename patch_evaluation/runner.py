from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from architectures import (
    processed_pixel_size_um,
    spatial_preprocessing_spec,
)
from modeling import (
    load_inference_model,
    resolve_device,
    validate_training_metadata,
)
from preprocess import build_inference_transform

from .data import build_test_data
from .inference import collect_predictions
from .integrated_gradients import run_integrated_gradients
from .metrics import calculate_binary_metrics
from .output import (
    append_result,
    metrics_to_dict,
    save_classification_report,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    save_predictions,
    save_roc_outputs,
    write_json,
)
from .settings import PatchEvaluationConfig


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_inputs(config: PatchEvaluationConfig) -> None:
    missing: list[str] = []
    if not config.test_dir.is_dir():
        missing.append(f"patch test directory: {config.test_dir}")
    if not Path(config.checkpoint).exists():
        missing.append(f"checkpoint: {config.checkpoint}")
    if config.target_image is not None and not config.target_image.is_file():
        missing.append(f"Reinhard target image: {config.target_image}")
    if missing:
        raise FileNotFoundError("Required input(s) not found:\n" + "\n".join(missing))


def run_patch_evaluation(config: PatchEvaluationConfig) -> dict:
    """Run deterministic patch inference, metrics, and optional IG figures."""
    torch.set_float32_matmul_precision("high")
    np.set_printoptions(suppress=True, precision=5)
    _seed_everything(config.seed)
    _validate_inputs(config)

    device = resolve_device(config.device)
    print(f"Experiment: {config.experiment}")
    print(f"Seed/fold: {config.seed}/{config.fold}")
    print(f"Device: {device}")
    print(f"Test directory: {config.test_dir}")
    print("Input normalization: ImageNet-1K mean/std")
    print(
        "Integrated Gradients: "
        + ("enabled" if config.integrated_gradients else "disabled (default)")
    )

    # Resolve the checkpoint architecture before constructing preprocessing;
    # ViT alone needs Resize(246) -> CenterCrop(224).
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
    data = build_test_data(
        config.test_dir,
        transform=transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )
    print(f"Test patches: {len(data.dataset)}")
    print(f"Classes: {data.classes}")

    first_path = Path(data.dataset.samples[0][0])
    with Image.open(first_path) as image_file:
        source_size = image_file.size
    expected_source_size = (config.source_patch_size, config.source_patch_size)
    if source_size != expected_source_size:
        raise ValueError(
            f"The first test patch {first_path} has source size {source_size}, "
            f"but --source-patch-size expects {expected_source_size}."
        )

    first_image, _, _, _ = data.dataset[0]
    actual_size = tuple(int(value) for value in first_image.shape[-2:])
    expected_model_size = (
        resolved_model_input_size,
        resolved_model_input_size,
    )
    if actual_size != expected_model_size:
        raise ValueError(
            f"Architecture-specific preprocessing produced {actual_size}, but "
            f"{architecture} expects {expected_model_size}."
        )

    training_metadata = validate_training_metadata(
        checkpoint_path,
        expected_experiment=config.experiment,
        expected_seed=config.seed,
        expected_fold=config.fold,
        expected_architecture=architecture,
        expected_input_size=resolved_model_input_size,
        expected_source_patch_size=config.source_patch_size,
    )
    model.to(device)
    model.eval()
    print(f"Architecture: {architecture}")
    print(f"Checkpoint: {checkpoint_path}")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = config.to_serializable_dict()
    run_config.update(
        {
            "resolved_architecture": architecture,
            "resolved_checkpoint": str(checkpoint_path),
            "source_patch_size": config.source_patch_size,
            "model_input_size": resolved_model_input_size,
            "spatial_preprocessing": spatial_preprocessing_spec(architecture),
            "resize_or_center_crop": architecture == "vit_b_16",
            "classes": list(data.classes),
            "test_patches": len(data.dataset),
            "training_run_config_found": training_metadata is not None,
        }
    )
    run_config["output_dir"] = str(output_dir)
    run_config["results_csv"] = str(results_csv)
    write_json(output_dir / "run_config.json", run_config)

    if config.validate_only:
        print("Validation completed; patch inference was not run.")
        return run_config

    predictions = collect_predictions(
        model,
        data.loader,
        device=device,
        amp=config.amp,
        num_classes=len(data.classes),
    )
    metrics = calculate_binary_metrics(
        predictions,
        classes=data.classes,
        positive_class=config.positive_class,
    )

    positive_index = data.classes.index(config.positive_class)
    negative_class = data.classes[1 - positive_index]
    display_classes = (negative_class, config.positive_class)

    save_predictions(
        output_dir / "patch_predictions.csv",
        predictions,
        classes=data.classes,
    )
    save_confusion_matrix_csv(
        output_dir / "confusion_matrix.csv",
        metrics.confusion_matrix,
        display_classes=display_classes,
    )
    save_confusion_matrix_plot(
        output_dir / "confusion_matrix.tiff",
        metrics.confusion_matrix,
        display_classes=display_classes,
        dpi=config.plot_dpi,
    )
    save_classification_report(
        output_dir / "classification_report.csv",
        output_dir / "classification_report.txt",
        metrics,
    )
    save_roc_outputs(
        output_dir / "roc_curve.csv",
        output_dir / "ROC_curve.tiff",
        metrics,
        positive_class=config.positive_class,
        dpi=config.plot_dpi,
    )
    metrics_payload = metrics_to_dict(
        metrics,
        classes=data.classes,
        positive_class=config.positive_class,
    )
    write_json(output_dir / "metrics.json", metrics_payload)

    correct = int(np.sum(predictions.labels == predictions.predictions))
    print(f"Accuracy: {metrics.accuracy:.4f} ({correct}/{predictions.size})")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics.confusion_matrix)
    print(metrics.classification_report_text)
    print(f"AUC: {metrics.auc:.4f}")
    print(f"Sensitivity: {metrics.sensitivity:.4f}")
    print(f"Specificity: {metrics.specificity:.4f}")

    ig_images = 0
    if config.integrated_gradients:
        # IG needs input gradients and therefore stays outside inference_mode
        # and CUDA autocast used by ordinary performance inference.
        ig_images = run_integrated_gradients(
            model=model,
            dataset=data.dataset,
            loader=data.loader,
            prediction_result=predictions,
            classes=data.classes,
            device=device,
            output_dir=output_dir / "integrated_gradients",
            baseline_type=config.ig_baseline,
            target_type=config.ig_target,
            n_steps=config.ig_n_steps,
            internal_batch_size=config.ig_internal_batch_size,
            overlay_alpha=config.ig_overlay_alpha,
            max_images=config.ig_max_images,
            only_misclassified=config.ig_only_misclassified,
            pixel_size_um=processed_pixel_size_um(
                architecture,
                source_patch_size=config.source_patch_size,
                source_pixel_size_um=config.pixel_size_um,
            ),
            scale_bar_um=config.scale_bar_um,
            dpi=config.ig_dpi,
        )
    else:
        print(
            "Integrated Gradients was skipped. Add --integrated-gradients "
            "to generate attribution figures."
        )

    summary = {
        "experiment": config.experiment,
        "stain": config.stain,
        "reinhard": config.use_reinhard,
        "seed": config.seed,
        "fold": config.fold,
        "architecture": architecture,
        "source_patch_size": config.source_patch_size,
        "model_input_size": resolved_model_input_size,
        "checkpoint": str(checkpoint_path),
        "test_dir": str(config.test_dir),
        "test_patches": predictions.size,
        **metrics_payload,
        "integrated_gradients": config.integrated_gradients,
        "ig_images": ig_images,
        "ig_baseline": config.ig_baseline if config.integrated_gradients else None,
        "ig_target": config.ig_target if config.integrated_gradients else None,
        "ig_n_steps": config.ig_n_steps if config.integrated_gradients else None,
        "ig_internal_batch_size": (
            config.ig_internal_batch_size if config.integrated_gradients else None
        ),
    }
    write_json(output_dir / "evaluation_summary.json", summary)

    append_result(
        results_csv,
        {
            "experiment": config.experiment,
            "stain": config.stain,
            "reinhard": config.use_reinhard,
            "seed": config.seed,
            "fold": config.fold,
            "architecture": architecture,
            "source_patch_size": config.source_patch_size,
            "model_input_size": resolved_model_input_size,
            "checkpoint": str(checkpoint_path),
            "test_dir": str(config.test_dir),
            "test_patches": predictions.size,
            "negative_class": negative_class,
            "positive_class": config.positive_class,
            "TN": metrics.tn,
            "FP": metrics.fp,
            "FN": metrics.fn,
            "TP": metrics.tp,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "AUC": metrics.auc,
            "accuracy": metrics.accuracy,
            "integrated_gradients": config.integrated_gradients,
            "ig_images": ig_images,
            "ig_baseline": (
                config.ig_baseline if config.integrated_gradients else None
            ),
            "ig_target": config.ig_target if config.integrated_gradients else None,
            "ig_n_steps": config.ig_n_steps if config.integrated_gradients else None,
            "ig_internal_batch_size": (
                config.ig_internal_batch_size
                if config.integrated_gradients
                else None
            ),
        },
    )

    print(f"Output directory: {output_dir}")
    print(f"Results CSV: {results_csv}")
    return summary
