from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .settings import WSIInferenceConfig


@dataclass(frozen=True)
class SamplingGeometry:
    """Pixel and physical geometry used to read and measure WSI tiles."""

    input_type: str
    slide_width_px: int
    slide_height_px: int
    input_mpp_x: float
    input_mpp_y: float
    mpp_source: str
    model_mpp: float
    model_tile_size_px: int
    model_stride_px: int
    source_tile_width_px: int
    source_tile_height_px: int
    source_stride_x_px: float
    source_stride_y_px: float
    model_tile_um: float
    model_stride_um: float
    grid_step_x_um: float
    grid_step_y_um: float
    resample_to_model_size: bool

    def to_serializable_dict(self) -> dict:
        return asdict(self)


def parse_metadata_mpp(raw_mpp) -> tuple[float, float] | None:
    if raw_mpp is None:
        return None
    try:
        values = np.asarray(raw_mpp, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    if values.size == 1:
        mpp_x = mpp_y = float(values[0])
    else:
        mpp_x, mpp_y = float(values[0]), float(values[1])
    if mpp_x <= 0 or mpp_y <= 0:
        return None
    return mpp_x, mpp_y


def resolve_input_mpp(
    wsi,
    override: tuple[float, float] | None,
) -> tuple[float, float, str]:
    metadata_mpp = parse_metadata_mpp(getattr(wsi.info, "mpp", None))
    if override is not None:
        mpp_x, mpp_y = map(float, override)
        if metadata_mpp is not None and max(
            abs(metadata_mpp[0] - mpp_x),
            abs(metadata_mpp[1] - mpp_y),
        ) > 0.001:
            print(
                "Warning: WSI metadata MPP "
                f"{metadata_mpp} differs from the command-line override "
                f"({mpp_x}, {mpp_y}); the override will be used."
            )
        return mpp_x, mpp_y, "command-line override"
    if metadata_mpp is None:
        raise ValueError(
            "Microns-per-pixel was not found in WSI metadata. "
            "Specify --input-mpp."
        )
    return metadata_mpp[0], metadata_mpp[1], "WSI metadata"


def _slide_dimensions(wsi) -> tuple[int, int]:
    dimensions = getattr(wsi.info, "slide_dimensions", None)
    if dimensions is None:
        raise ValueError("WSI metadata does not contain slide_dimensions.")
    values = np.asarray(dimensions, dtype=np.int64).reshape(-1)
    if values.size < 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"Invalid slide_dimensions: {dimensions}")
    return int(values[0]), int(values[1])


def build_sampling_geometry(
    wsi,
    config: WSIInferenceConfig,
) -> SamplingGeometry:
    mpp_x, mpp_y, mpp_source = resolve_input_mpp(
        wsi,
        config.input_mpp_override,
    )
    slide_width, slide_height = _slide_dimensions(wsi)
    model_tile_um = config.tile_size * config.model_mpp
    model_stride_um = config.stride * config.model_mpp

    if config.input_type == "post-xenium":
        source_tile_width_float = model_tile_um / mpp_x
        source_tile_height_float = model_tile_um / mpp_y
        source_stride_x = model_stride_um / mpp_x
        source_stride_y = model_stride_um / mpp_y
        source_tile_width = max(1, int(round(source_tile_width_float)))
        source_tile_height = max(1, int(round(source_tile_height_float)))
        grid_step_x_um = model_stride_um
        grid_step_y_um = model_stride_um
        resample = (
            source_tile_width != config.tile_size
            or source_tile_height != config.tile_size
        )
    else:
        source_tile_width = config.tile_size
        source_tile_height = config.tile_size
        source_stride_x = float(config.stride)
        source_stride_y = float(config.stride)
        grid_step_x_um = source_stride_x * mpp_x
        grid_step_y_um = source_stride_y * mpp_y
        resample = False

    return SamplingGeometry(
        input_type=config.input_type,
        slide_width_px=slide_width,
        slide_height_px=slide_height,
        input_mpp_x=mpp_x,
        input_mpp_y=mpp_y,
        mpp_source=mpp_source,
        model_mpp=config.model_mpp,
        model_tile_size_px=config.tile_size,
        model_stride_px=config.stride,
        source_tile_width_px=source_tile_width,
        source_tile_height_px=source_tile_height,
        source_stride_x_px=source_stride_x,
        source_stride_y_px=source_stride_y,
        model_tile_um=model_tile_um,
        model_stride_um=model_stride_um,
        grid_step_x_um=grid_step_x_um,
        grid_step_y_um=grid_step_y_um,
        resample_to_model_size=resample,
    )


def make_axis_positions(
    minimum: float,
    maximum: float,
    *,
    tile_span_px: int,
    stride_px: float,
    slide_length_px: int,
) -> np.ndarray:
    """Create integer level-0 positions while retaining a fractional stride."""
    if tile_span_px <= 0 or stride_px <= 0 or slide_length_px <= 0:
        raise ValueError("Invalid grid geometry.")
    if tile_span_px > slide_length_px:
        raise ValueError(
            f"Tile span ({tile_span_px} px) exceeds slide dimension "
            f"({slide_length_px} px)."
        )

    start = max(0, int(math.floor(minimum)))
    last = min(
        slide_length_px - tile_span_px,
        int(math.ceil(maximum)) - tile_span_px,
    )

    if last < start:
        centre = 0.5 * (minimum + maximum)
        centred = int(round(centre - tile_span_px / 2.0))
        centred = min(max(0, centred), slide_length_px - tile_span_px)
        return np.asarray([centred], dtype=np.int64)

    count = int(math.floor((last - start) / stride_px)) + 1
    positions = [int(round(start + index * stride_px)) for index in range(count)]
    return np.asarray(list(dict.fromkeys(positions)), dtype=np.int64)


def make_grid(
    bounds: tuple[float, float, float, float],
    geometry: SamplingGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = bounds
    xs = make_axis_positions(
        minx,
        maxx,
        tile_span_px=geometry.source_tile_width_px,
        stride_px=geometry.source_stride_x_px,
        slide_length_px=geometry.slide_width_px,
    )
    ys = make_axis_positions(
        miny,
        maxy,
        tile_span_px=geometry.source_tile_height_px,
        stride_px=geometry.source_stride_y_px,
        slide_length_px=geometry.slide_height_px,
    )
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("No WSI inference grid positions were generated.")
    return xs, ys
