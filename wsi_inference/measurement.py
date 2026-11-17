from __future__ import annotations

from dataclasses import dataclass
import inspect
import math

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

from .sampling import SamplingGeometry


COMPONENT_COLUMNS = [
    "label",
    "area_mm2",
    "major_axis_mm",
    "farthest_mm",
    "bbox_long_mm",
    "centroid_row",
    "centroid_col",
]


@dataclass(frozen=True)
class MeasurementResult:
    binary_mask: np.ndarray
    label_map: np.ndarray
    components: pd.DataFrame
    grid_step_x_um: float
    grid_step_y_um: float
    pixel_area_mm2: float
    min_area_px: int

    @property
    def pixel_size_um(self) -> float:
        return 0.5 * (self.grid_step_x_um + self.grid_step_y_um)

    @property
    def invasion_size_mm(self) -> float:
        if self.components.empty:
            return 0.0
        return float(self.components.iloc[0]["farthest_mm"])


def _convex_hull_diameter(points_mm: np.ndarray) -> float:
    if len(points_mm) < 2:
        return 0.0
    if len(points_mm) > 2:
        try:
            points_mm = points_mm[ConvexHull(points_mm).vertices]
        except QhullError:
            centered = points_mm - points_mm.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ vh[0]
            return float(projection.max() - projection.min())
    return float(pdist(points_mm).max())


def farthest_distance_mm(
    region_mask: np.ndarray,
    *,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    input_mpp_x: float | None = None,
    input_mpp_y: float | None = None,
    pixel_size_um: float | None = None,
) -> float:
    rows, columns = np.nonzero(region_mask)
    if len(rows) < 2:
        return 0.0

    if xs is not None and ys is not None:
        if input_mpp_x is None or input_mpp_y is None:
            raise ValueError("input_mpp_x and input_mpp_y are required with xs/ys.")
        x_mm = np.asarray(xs, dtype=np.float64)[columns] * input_mpp_x / 1000.0
        y_mm = np.asarray(ys, dtype=np.float64)[rows] * input_mpp_y / 1000.0
        points_mm = np.column_stack((x_mm, y_mm))
    else:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um or xs/ys with input MPP is required.")
        points_mm = (
            np.column_stack((columns, rows)).astype(np.float64)
            * pixel_size_um
            / 1000.0
        )
    return _convex_hull_diameter(points_mm)


def _remove_small_objects_compat(
    binary_mask: np.ndarray,
    min_area_px: int,
) -> np.ndarray:
    if "max_size" in inspect.signature(remove_small_objects).parameters:
        return remove_small_objects(
            binary_mask,
            max_size=max(0, min_area_px - 1),
        )
    return remove_small_objects(binary_mask, min_size=min_area_px)


def postprocess_probability_map(
    probability_map: np.ndarray,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    sampling_geometry: SamplingGeometry,
    invasion_threshold: float,
    min_area_mm2: float,
) -> MeasurementResult:
    if probability_map.ndim != 2:
        raise ValueError("probability_map must be a two-dimensional array.")
    if probability_map.shape != (len(ys), len(xs)):
        raise ValueError(
            "probability_map shape does not match the supplied grid coordinates."
        )

    filled = np.nan_to_num(
        probability_map,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    binary_mask = filled >= invasion_threshold
    try:
        binary_mask = closing(binary_mask, footprint=disk(1), mode="ignore")
    except TypeError:
        binary_mask = closing(binary_mask, footprint=disk(1))

    pixel_area_mm2 = (
        sampling_geometry.grid_step_x_um
        * sampling_geometry.grid_step_y_um
        / 1e6
    )
    if sampling_geometry.input_type == "post-xenium":
        min_area_px = max(1, int(math.ceil(min_area_mm2 / pixel_area_mm2)))
    else:
        # Retain the integer conversion used by the conventional source scripts.
        min_area_px = max(1, int(min_area_mm2 / pixel_area_mm2))
    binary_mask = _remove_small_objects_compat(binary_mask, min_area_px)

    label_map = label(binary_mask)
    raw_regions = regionprops(label_map)
    try:
        physical_regions = {
            region.label: region
            for region in regionprops(
                label_map,
                spacing=(
                    sampling_geometry.grid_step_y_um / 1000.0,
                    sampling_geometry.grid_step_x_um / 1000.0,
                ),
            )
        }
    except TypeError:
        physical_regions = {}

    rows: list[dict[str, float | int]] = []
    for region in raw_regions:
        region_mask = label_map == region.label
        min_row, min_col, max_row, max_col = region.bbox

        if sampling_geometry.input_type == "post-xenium":
            bbox_x0 = int(xs[min_col])
            bbox_y0 = int(ys[min_row])
            bbox_x1 = (
                int(xs[max_col - 1])
                + sampling_geometry.source_tile_width_px
            )
            bbox_y1 = (
                int(ys[max_row - 1])
                + sampling_geometry.source_tile_height_px
            )
            bbox_width_mm = (
                (bbox_x1 - bbox_x0)
                * sampling_geometry.input_mpp_x
                / 1000.0
            )
            bbox_height_mm = (
                (bbox_y1 - bbox_y0)
                * sampling_geometry.input_mpp_y
                / 1000.0
            )
        else:
            bbox_width_mm = (
                (max_col - min_col)
                * sampling_geometry.grid_step_x_um
                / 1000.0
            )
            bbox_height_mm = (
                (max_row - min_row)
                * sampling_geometry.grid_step_y_um
                / 1000.0
            )

        physical_region = physical_regions.get(region.label)
        if physical_region is not None:
            major_axis_mm = float(physical_region.axis_major_length)
        else:
            major_axis_mm = float(
                region.axis_major_length
                * 0.5
                * (
                    sampling_geometry.grid_step_x_um
                    + sampling_geometry.grid_step_y_um
                )
                / 1000.0
            )

        rows.append(
            {
                "label": int(region.label),
                "area_mm2": float(region.area * pixel_area_mm2),
                "major_axis_mm": major_axis_mm,
                "farthest_mm": farthest_distance_mm(
                    region_mask,
                    xs=xs,
                    ys=ys,
                    input_mpp_x=sampling_geometry.input_mpp_x,
                    input_mpp_y=sampling_geometry.input_mpp_y,
                ),
                "bbox_long_mm": float(max(bbox_width_mm, bbox_height_mm)),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
            }
        )

    components = pd.DataFrame(rows, columns=COMPONENT_COLUMNS)
    if not components.empty:
        components = components.sort_values(
            "farthest_mm",
            ascending=False,
        ).reset_index(drop=True)

    return MeasurementResult(
        binary_mask=np.asarray(binary_mask, dtype=bool),
        label_map=np.asarray(label_map),
        components=components,
        grid_step_x_um=sampling_geometry.grid_step_x_um,
        grid_step_y_um=sampling_geometry.grid_step_y_um,
        pixel_area_mm2=pixel_area_mm2,
        min_area_px=min_area_px,
    )
