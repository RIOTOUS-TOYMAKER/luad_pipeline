from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .sampling import SamplingGeometry, make_grid


@dataclass(frozen=True)
class TileInferenceResult:
    probability_map: np.ndarray
    evaluated_mask: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    grid_tiles: int
    roi_tiles: int
    tissue_tiles: int


def _as_uint8_rgb(tile: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile)
    if tile.ndim == 2:
        tile = np.repeat(tile[..., None], 3, axis=2)
    if tile.ndim != 3 or tile.shape[2] < 3:
        raise ValueError(f"WSIReader returned an unexpected tile shape: {tile.shape}")
    tile = tile[..., :3]

    if tile.dtype == np.uint8:
        return np.ascontiguousarray(tile)
    if np.issubdtype(tile.dtype, np.integer):
        maximum = float(np.iinfo(tile.dtype).max)
        tile = np.rint(tile.astype(np.float32) * (255.0 / maximum))
    elif np.issubdtype(tile.dtype, np.floating):
        tile = tile.astype(np.float32)
        finite = tile[np.isfinite(tile)]
        if finite.size == 0:
            raise ValueError("WSIReader returned a tile with no finite pixel values.")
        if finite.min() >= 0.0 and finite.max() <= 1.0:
            tile = tile * 255.0
    else:
        raise ValueError(f"Unsupported WSI tile dtype: {tile.dtype}")
    return np.ascontiguousarray(np.clip(tile, 0, 255).astype(np.uint8))


def _resize_interpolation(width: int, height: int, target_size: int) -> int:
    if width < target_size or height < target_size:
        return cv2.INTER_CUBIC
    if width > target_size or height > target_size:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def read_model_tile(
    *,
    wsi,
    x: int,
    y: int,
    geometry: SamplingGeometry,
) -> np.ndarray:
    kwargs = {
        "location": (int(x), int(y)),
        "size": (
            int(geometry.source_tile_width_px),
            int(geometry.source_tile_height_px),
        ),
        "resolution": 0,
        "units": "level",
        "coord_space": "baseline",
    }
    try:
        tile = wsi.read_rect(**kwargs)
    except TypeError:
        # Compatibility with TIAToolbox releases whose read_rect did not expose
        # coord_space as a keyword.
        kwargs.pop("coord_space")
        tile = wsi.read_rect(**kwargs)

    tile = _as_uint8_rgb(tile)
    target_size = geometry.model_tile_size_px
    if tile.shape[0] != target_size or tile.shape[1] != target_size:
        tile = cv2.resize(
            tile,
            (target_size, target_size),
            interpolation=_resize_interpolation(
                tile.shape[1], tile.shape[0], target_size
            ),
        )
    return _as_uint8_rgb(tile)


def _tissue_ratio(tile_rgb: np.ndarray, saturation_threshold: int) -> float:
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    return float((hsv[..., 1] > saturation_threshold).mean())


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def infer_probability_map(
    *,
    wsi,
    roi_geometry: BaseGeometry,
    sampling_geometry: SamplingGeometry,
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    roi_overlap_threshold: float,
    saturation_threshold: int,
    min_tissue_ratio: float,
    invasion_class_index: int,
    amp: bool,
) -> TileInferenceResult:
    xs, ys = make_grid(roi_geometry.bounds, sampling_geometry)
    probability_map = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    evaluated_mask = np.zeros((len(ys), len(xs)), dtype=bool)

    prepared_roi = prep(roi_geometry)
    tile_area = float(
        sampling_geometry.source_tile_width_px
        * sampling_geometry.source_tile_height_px
    )
    pending_tiles: list[torch.Tensor] = []
    pending_coords: list[tuple[int, int]] = []
    roi_tiles = 0
    tissue_tiles = 0

    def flush_batch() -> None:
        nonlocal pending_tiles, pending_coords
        if not pending_tiles:
            return
        batch = torch.stack(pending_tiles).to(device, non_blocking=True)
        with _autocast_context(device, amp):
            logits = model(batch)
            if logits.ndim != 2 or invasion_class_index >= logits.shape[1]:
                raise RuntimeError(
                    "Model output does not contain the requested invasion class: "
                    f"shape={tuple(logits.shape)}, index={invasion_class_index}."
                )
            probabilities = torch.softmax(logits, dim=1)[:, invasion_class_index]
        for (row, column), probability in zip(
            pending_coords,
            probabilities.detach().float().cpu().numpy(),
            strict=True,
        ):
            probability_map[row, column] = float(probability)
            evaluated_mask[row, column] = True
        pending_tiles = []
        pending_coords = []

    with torch.inference_mode():
        for row, y in enumerate(ys):
            for column, x in enumerate(xs):
                tile_polygon = box(
                    int(x),
                    int(y),
                    int(x) + sampling_geometry.source_tile_width_px,
                    int(y) + sampling_geometry.source_tile_height_px,
                )
                if not prepared_roi.intersects(tile_polygon):
                    continue
                overlap = tile_polygon.intersection(roi_geometry).area / tile_area
                if overlap < roi_overlap_threshold:
                    continue
                roi_tiles += 1

                tile_rgb = read_model_tile(
                    wsi=wsi,
                    x=int(x),
                    y=int(y),
                    geometry=sampling_geometry,
                )
                if _tissue_ratio(tile_rgb, saturation_threshold) < min_tissue_ratio:
                    continue
                tissue_tiles += 1

                pending_tiles.append(
                    transform(Image.fromarray(tile_rgb, mode="RGB"))
                )
                pending_coords.append((row, column))
                if len(pending_tiles) >= batch_size:
                    flush_batch()

        flush_batch()

    return TileInferenceResult(
        probability_map=probability_map,
        evaluated_mask=evaluated_mask,
        xs=xs,
        ys=ys,
        grid_tiles=int(len(xs) * len(ys)),
        roi_tiles=roi_tiles,
        tissue_tiles=tissue_tiles,
    )
