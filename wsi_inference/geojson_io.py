from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _iter_geometry_mappings(data: object) -> Iterable[dict]:
    """Yield geometry mappings from QuPath lists and standard GeoJSON."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_geometry_mappings(item)
        return

    if not isinstance(data, dict):
        raise ValueError("GeoJSON root must be a dictionary or a list.")

    object_type = data.get("type")

    if object_type == "FeatureCollection":
        for feature in data.get("features", []):
            yield from _iter_geometry_mappings(feature)
        return

    if object_type == "Feature" or "geometry" in data:
        geometry = data.get("geometry")
        if geometry is not None:
            if not isinstance(geometry, dict):
                raise ValueError("A GeoJSON feature contains an invalid geometry.")
            yield geometry
        return

    # A raw Polygon/MultiPolygon/GeometryCollection mapping.
    if object_type is not None:
        yield data
        return

    raise ValueError("Unknown GeoJSON structure.")


def _repair_if_needed(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_valid:
        return geometry

    try:
        from shapely import make_valid

        repaired = make_valid(geometry)
    except (ImportError, AttributeError):
        repaired = geometry.buffer(0)

    return repaired


def load_roi_geometry(path: Path) -> BaseGeometry:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tumor ROI GeoJSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    geometries: list[BaseGeometry] = []
    for mapping in _iter_geometry_mappings(data):
        geometry = _repair_if_needed(shape(mapping))
        if not geometry.is_empty and geometry.area > 0:
            geometries.append(geometry)

    if not geometries:
        raise ValueError(f"No non-empty polygonal ROI was found in {path}.")

    union = _repair_if_needed(unary_union(geometries))
    if union.is_empty or union.area <= 0:
        raise ValueError(f"The union of tumor ROIs is empty: {path}")

    return union
