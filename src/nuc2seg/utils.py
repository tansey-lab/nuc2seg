import os
import re
import logging

import geopandas
import numpy as np
import shapely
from blended_tiling import TilingModule
from shapely import box, Polygon
from shapely.geometry import box
import torch

logger = logging.getLogger(__name__)


def sqexp(x1, x2, bandwidth=2, scale=1, axis=None):
    return scale * np.exp(-np.linalg.norm(x1 - x2, axis=axis) ** 2 / bandwidth**2)


def grid_graph_edges(rows, cols):
    from collections import defaultdict

    edges = defaultdict(list)
    for x in range(cols):
        for y in range(rows):
            if x < cols - 1:
                i = int(y * cols + x)
                j = int(y * cols + x + 1)
                edges[i].append(j)
                edges[j].append(i)
            if y < rows - 1:
                i = int(y * cols + x)
                j = int((y + 1) * cols + x)
                edges[i].append(j)
                edges[j].append(i)
    return edges


def get_indices_for_ndarray(x_dim, y_dim):
    xy = np.mgrid[0:x_dim, 0:y_dim]
    return np.array(list(zip(xy[0].flatten(), xy[1].flatten())))


def get_tile_idx(fn):
    fn_clean = os.path.splitext(os.path.basename(fn))[0]
    # search for `tile_{number}` and extract number with regex
    return int(re.search(r"tile_(\d+)", fn_clean).group(1))


def get_indexed_tiles(extent, tile_size, overlap):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=overlap,
        base_size=extent,
    )

    result = {}
    for idx, (x1, y1, x2, y2) in enumerate(
        generate_tiles(
            tiler,
            x_extent=extent[0],
            y_extent=extent[1],
            overlap_fraction=overlap,
            tile_size=tile_size,
        )
    ):
        result[idx] = (x1, y1, x2, y2)
    return result


def get_tile_ids_for_bbox(extent, tile_size, overlap, bbox: Polygon):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=overlap,
        base_size=extent,
    )

    result = []
    for idx, (x1, y1, x2, y2) in enumerate(
        generate_tiles(
            tiler,
            x_extent=extent[0],
            y_extent=extent[1],
            overlap_fraction=overlap,
            tile_size=tile_size,
        )
    ):
        if box(x1, y1, x2, y2).intersects(bbox):
            result.append(idx)
    return result


def generate_tiles(
    tiler: TilingModule, x_extent, y_extent, tile_size, overlap_fraction, tile_ids=None
):
    """
    A generator function to yield overlapping tiles

    Yields:
    - BBox extent in pixels for each tile (non inclusive end) x1, y1, x2, y2
    """
    # Generate tiles
    tile_id = 0
    for x in tiler._calc_tile_coords(x_extent, tile_size[0], overlap_fraction)[0]:
        for y in tiler._calc_tile_coords(y_extent, tile_size[1], overlap_fraction)[0]:
            if tile_ids is not None:
                if tile_id in tile_ids:
                    yield x, y, x + tile_size[0], y + tile_size[1]
            else:
                yield x, y, x + tile_size[0], y + tile_size[1]
            tile_id += 1


def spatial_join_polygons_and_transcripts(
    boundaries: geopandas.GeoDataFrame, transcripts: geopandas.GeoDataFrame
):
    joined_gdf = geopandas.sjoin(boundaries, transcripts, how="inner")

    return joined_gdf


def drop_invalid_geometries(gdf: geopandas.GeoDataFrame):
    """
    Remove rows with invalid geometries from a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame

    Returns:
    GeoDataFrame: Clean GeoDataFrame with only valid geometries
    """
    # Create mask of valid geometries
    valid_mask = gdf.geometry.is_valid

    # Get count of invalid geometries
    invalid_count = (~valid_mask).sum()

    # Drop invalid geometries
    clean_gdf = gdf[valid_mask].copy()

    # Reset index after dropping rows
    clean_gdf.reset_index(drop=True, inplace=True)

    logger.info(f"Removed {invalid_count} invalid geometries")
    return clean_gdf


def create_shapely_rectangle(x1, y1, x2, y2):
    return box(x1, y1, x2, y2)


def filter_gdf_to_intersects_polygon(gdf, polygon=None):
    if polygon is None:
        return gdf
    return gdf[gdf.geometry.intersects(polygon)]


def filter_gdf_to_inside_polygon(gdf, polygon=None):
    if gdf.empty:
        return gdf

    if polygon is None:
        return gdf
    return gdf[gdf.geometry.apply(lambda x: polygon.contains(x))]


def get_tile_bounds(
    tile_width: int,
    tile_height: int,
    tile_overlap: float,
    tile_index: int,
    base_width: int,
    base_height: int,
) -> tuple[int, int, int, int]:
    tiler = TilingModule(
        tile_size=(tile_width, tile_height),
        tile_overlap=(tile_overlap, tile_overlap),
        base_size=(base_width, base_height),
    )
    x1, y1, x2, y2 = next(
        generate_tiles(
            tiler=tiler,
            x_extent=base_width,
            y_extent=base_height,
            tile_size=(tile_width, tile_height),
            overlap_fraction=tile_overlap,
            tile_ids=[tile_index],
        )
    )

    return (x1, y1, x2, y2)


def buffer_gdf(
    gdf: geopandas.GeoDataFrame, distance: float, inplace: bool = False
) -> geopandas.GeoDataFrame | None:
    """
    Expands all polygons in a GeoDataFrame by a specified distance.

    Args:
        gdf: Input GeoDataFrame
        distance: Buffer distance (positive for expansion, negative for shrinking)
        inplace: If True, modifies the input GeoDataFrame, else returns a copy

    Returns:
        Modified GeoDataFrame if inplace=False, else None
    """
    if not inplace:
        gdf = gdf.copy()

    gdf["geometry"] = gdf.geometry.buffer(
        distance, cap_style="round", join_style="round"
    )

    return None if inplace else gdf


def create_torch_polygon(polygon: shapely.Polygon, device) -> torch.Tensor:
    """
    Create a polygon tensor from a list of vertex coordinates.

    Args:
        polygon: shapely polygon

    Returns:
        Tensor of shape (N, 2) containing the polygon vertices
    """
    if polygon.boundary.is_ring:

        vertices = list(
            zip(polygon.boundary.coords.xy[0], polygon.boundary.coords.xy[1])
        )
    else:
        if polygon.boundary.geom_type == "MultiLineString":
            longest_line = max(
                list(polygon.boundary.geoms), key=lambda x: len(x.coords)
            )
            if longest_line.is_ring:
                vertices = list(
                    zip(longest_line.coords.xy[0], longest_line.coords.xy[1])
                )
        else:
            raise ValueError

    return torch.tensor(vertices, dtype=torch.float32, device=device)
