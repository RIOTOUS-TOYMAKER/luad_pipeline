from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .measurement import MeasurementResult
from .sampling import SamplingGeometry
from .settings import WSIInferenceConfig
from .tiling import TileInferenceResult


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_array_outputs(
    output_dir: Path,
    tile_result: TileInferenceResult,
    measurement: MeasurementResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "invasion_prob_map.npy",
        np.nan_to_num(tile_result.probability_map, nan=0.0),
    )
    np.save(output_dir / "evaluated_tile_mask.npy", tile_result.evaluated_mask)
    np.save(output_dir / "invasion_binary_mask.npy", measurement.binary_mask)
    np.savez_compressed(
        output_dir / "grid_coordinates.npz",
        xs=tile_result.xs,
        ys=tile_result.ys,
    )


def _iter_polygons(geometry: BaseGeometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _iter_polygons(child)


def _repair(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_valid:
        return geometry
    try:
        from shapely import make_valid

        return make_valid(geometry)
    except (ImportError, AttributeError):
        return geometry.buffer(0)


def _polygon_exterior_ring(
    polygon: Polygon,
    *,
    decimals: int = 3,
) -> list[list[float]]:
    ring = [
        [round(float(x), decimals), round(float(y), decimals)]
        for x, y in polygon.exterior.coords
    ]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def make_invasion_geojson(
    measurement: MeasurementResult,
    *,
    tile_result: TileInferenceResult,
    sampling_geometry: SamplingGeometry,
    roi_geometry: BaseGeometry,
    config: WSIInferenceConfig,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    component_lookup = {
        int(row["label"]): row
        for row in measurement.components.to_dict(orient="records")
    }
    simplify_px = 0.0
    if config.geojson_simplify_um > 0:
        mean_mpp = 0.5 * (
            sampling_geometry.input_mpp_x + sampling_geometry.input_mpp_y
        )
        simplify_px = config.geojson_simplify_um / mean_mpp

    for component_label, metrics in component_lookup.items():
        rows, columns = np.nonzero(measurement.label_map == component_label)
        tile_polygons = [
            box(
                int(tile_result.xs[column]),
                int(tile_result.ys[row]),
                int(tile_result.xs[column])
                + sampling_geometry.source_tile_width_px,
                int(tile_result.ys[row])
                + sampling_geometry.source_tile_height_px,
            )
            for row, column in zip(rows, columns, strict=True)
        ]
        if not tile_polygons:
            continue
        merged = _repair(unary_union(tile_polygons))
        if config.resolved_clip_output_to_roi:
            merged = _repair(merged.intersection(roi_geometry))
        if simplify_px > 0 and not merged.is_empty:
            merged = _repair(
                merged.simplify(simplify_px, preserve_topology=True)
            )

        for polygon_number, polygon in enumerate(
            _iter_polygons(merged),
            start=1,
        ):
            if polygon.is_empty or polygon.area <= 0:
                continue
            properties = {
                "name": f"invasion_{component_label}",
                "label": component_label,
                "polygon_number": polygon_number,
                "area_mm2": float(metrics["area_mm2"]),
                "major_axis_mm": float(metrics["major_axis_mm"]),
                "farthest_mm": float(metrics["farthest_mm"]),
                "bbox_long_mm": float(metrics["bbox_long_mm"]),
            }

            if config.resolved_geojson_profile == "xenium":
                # Preserve exterior rings only, following the post-Xenium source
                # script's Xenium annotation export convention.
                ring = _polygon_exterior_ring(polygon)
                if len(ring) < 4:
                    continue
                if len(ring) >= 100_000:
                    raise RuntimeError(
                        "A GeoJSON polygon contains 100,000 or more vertices. "
                        "Increase --geojson-simplify-um."
                    )
                properties["name"] = (
                    f"AI_invasion_{component_label}_{polygon_number}"
                )
                properties.update(
                    {
                        "objectType": "annotation",
                        "classification": {
                            "name": config.classification_name,
                            "color": list(config.classification_color),
                        },
                        "sourceImage": config.wsi_path.name,
                        "coordinateSpace": config.coordinate_space_description,
                    }
                )
                features.append(
                    {
                        "type": "Feature",
                        "id": str(uuid4()),
                        "properties": properties,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [ring],
                        },
                    }
                )
            else:
                features.append(
                    {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": mapping(polygon),
                    }
                )

    if config.resolved_geojson_profile == "xenium" and len(features) >= 10_000:
        raise RuntimeError(
            "The output contains 10,000 or more polygons. Increase "
            "--geojson-simplify-um or revise the postprocessing settings."
        )
    return {"type": "FeatureCollection", "features": features}


def append_legacy_result(path: Path, *, case: int, invasion_size_mm: float) -> None:
    """Preserve the source CSV schema: case,invasion_size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["case", "invasion_size"])
        writer.writerow([case, invasion_size_mm])
