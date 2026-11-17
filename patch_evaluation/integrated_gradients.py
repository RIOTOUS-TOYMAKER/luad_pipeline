from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from torch.utils.data import DataLoader

from preprocess import denormalize_imagenet1k

from .data import ImageFolderWithPaths
from .inference import PredictionResult


def _load_integrated_gradients_class():
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "Integrated Gradients was requested, but Captum is not installed. "
            "Install it with: python -m pip install 'captum>=0.7'"
        ) from exc
    return IntegratedGradients


def _mean_transformed_image(loader: DataLoader) -> torch.Tensor:
    running_sum: torch.Tensor | None = None
    count = 0
    for images, _labels, _paths, _indices in loader:
        batch_sum = images.to(dtype=torch.float64).sum(dim=0)
        running_sum = batch_sum if running_sum is None else running_sum + batch_sum
        count += int(images.shape[0])
    if running_sum is None or count == 0:
        raise ValueError("Cannot calculate an IG mean baseline from an empty loader.")
    return (running_sum / count).to(dtype=torch.float32)


def _add_scalebar(
    axis,
    *,
    pixel_size_um: float,
    bar_length_um: float,
) -> None:
    bar_pixels = bar_length_um / pixel_size_um
    scale_bar = AnchoredSizeBar(
        axis.transData,
        bar_pixels,
        f"{bar_length_um:g} μm",
        "lower right",
        pad=0.5,
        color="white",
        frameon=False,
        size_vertical=2,
        fontproperties=fm.FontProperties(size=10),
    )
    axis.add_artist(scale_bar)


def _safe_stem(path: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9._+()=-]+",
        "-",
        Path(path).stem,
    ).strip("-")
    return cleaned or "patch"


def _selected_indices(
    prediction_result: PredictionResult,
    *,
    only_misclassified: bool,
    max_images: int | None,
) -> list[int]:
    indices = list(range(prediction_result.size))
    if only_misclassified:
        indices = [
            index
            for index in indices
            if prediction_result.labels[index]
            != prediction_result.predictions[index]
        ]
    return indices if max_images is None else indices[:max_images]


def run_integrated_gradients(
    *,
    model: torch.nn.Module,
    dataset: ImageFolderWithPaths,
    loader: DataLoader,
    prediction_result: PredictionResult,
    classes: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
    baseline_type: str,
    target_type: str,
    n_steps: int,
    internal_batch_size: int,
    overlay_alpha: float,
    max_images: int | None,
    only_misclassified: bool,
    pixel_size_um: float,
    scale_bar_um: float,
    dpi: int,
) -> int:
    """Generate per-patch IG figures and one complete convergence-delta CSV."""
    IntegratedGradients = _load_integrated_gradients_class()
    selected = _selected_indices(
        prediction_result,
        only_misclassified=only_misclassified,
        max_images=max_images,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = output_dir / "convergence_delta.csv"
    fieldnames = [
        "sample_index",
        "filename",
        "path",
        "true_label",
        "predicted_label",
        "target_label",
        "baseline",
        "n_steps",
        "convergence_delta",
    ]

    if not selected:
        with convergence_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        print("Integrated Gradients: no patches matched the selection criteria.")
        return 0

    mean_image = (
        _mean_transformed_image(loader)
        if baseline_type == "mean"
        else None
    )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attribution_method = IntegratedGradients(model)

    with convergence_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for completed, result_index in enumerate(selected, start=1):
            image, dataset_label, path, dataset_index = dataset[result_index]
            if dataset_index != result_index:
                raise RuntimeError("Dataset index changed during IG generation.")

            true_index = int(prediction_result.labels[result_index])
            predicted_index = int(prediction_result.predictions[result_index])
            if int(dataset_label) != true_index or path != prediction_result.paths[result_index]:
                raise RuntimeError(
                    "Dataset order changed between prediction and IG generation."
                )

            target_index = (
                predicted_index if target_type == "predicted" else true_index
            )
            input_tensor = image.unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)
            if baseline_type == "zero":
                baseline = torch.zeros_like(input_tensor)
            else:
                assert mean_image is not None
                baseline = mean_image.unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)
            attribution, delta = attribution_method.attribute(
                input_tensor,
                baselines=baseline,
                target=target_index,
                return_convergence_delta=True,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )

            heatmap = (
                attribution.detach()
                .abs()
                .mean(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            visible_image = (
                denormalize_imagenet1k(image)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            true_name = classes[true_index]
            predicted_name = classes[predicted_index]
            target_name = classes[target_index]
            title_color = "black" if true_index == predicted_index else "blue"

            figure, axes = plt.subplots(
                1,
                4,
                figsize=(12, 4),
                gridspec_kw={"width_ratios": [1, 1, 1, 0.04]},
                constrained_layout=True,
            )
            original_axis, overlay_axis, heatmap_axis, colorbar_axis = axes

            original_axis.imshow(visible_image)
            original_axis.set_title(
                f"True: {true_name}\nPred: {predicted_name}",
                color=title_color,
                fontsize=16,
            )
            original_axis.axis("off")
            _add_scalebar(
                original_axis,
                pixel_size_um=pixel_size_um,
                bar_length_um=scale_bar_um,
            )

            overlay_axis.imshow(visible_image)
            overlay_axis.imshow(heatmap, cmap="jet", alpha=overlay_alpha)
            overlay_axis.set_title("Overlay", fontsize=16)
            overlay_axis.axis("off")
            _add_scalebar(
                overlay_axis,
                pixel_size_um=pixel_size_um,
                bar_length_um=scale_bar_um,
            )

            heatmap_image = heatmap_axis.imshow(heatmap, cmap="jet")
            heatmap_axis.set_title(f"IG target={target_name}", fontsize=16)
            heatmap_axis.axis("off")
            _add_scalebar(
                heatmap_axis,
                pixel_size_um=pixel_size_um,
                bar_length_um=scale_bar_um,
            )
            figure.colorbar(heatmap_image, cax=colorbar_axis)

            output_name = (
                f"{result_index:05d}_{_safe_stem(path)}"
                f"_true-{true_name}_pred-{predicted_name}"
                f"_target-{target_name}.tiff"
            )
            figure.savefig(
                output_dir / output_name,
                bbox_inches="tight",
                pad_inches=0.02,
                dpi=dpi,
            )
            plt.close(figure)

            writer.writerow(
                {
                    "sample_index": result_index,
                    "filename": Path(path).name,
                    "path": path,
                    "true_label": true_name,
                    "predicted_label": predicted_name,
                    "target_label": target_name,
                    "baseline": baseline_type,
                    "n_steps": n_steps,
                    "convergence_delta": float(delta.detach().mean().cpu()),
                }
            )
            file.flush()

            del attribution, delta, input_tensor, baseline
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"Integrated Gradients: {completed}/{len(selected)} "
                f"({Path(path).name})"
            )

    return len(selected)
