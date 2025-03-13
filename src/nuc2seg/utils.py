import math
import os
import re
import logging
from typing import Optional

import anndata
import geopandas
import numpy as np
import pandas as pd
import shapely
from blended_tiling import TilingModule
from shapely import box, Polygon
from shapely.geometry import box
from shapely.affinity import translate, scale
import torch
from shapely.ops import linemerge, unary_union, polygonize, clip_by_rect
from shapely.geometry import LineString, Polygon


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


def safe_squeeze(arr):
    squeezed = arr.squeeze()

    if squeezed.shape == ():
        return np.array([squeezed])
    else:
        return squeezed


def filter_anndata_to_sample_area(adata: anndata.AnnData, sample_area: shapely.Polygon):
    centroids = adata.obsm["spatial"]
    points = [shapely.geometry.Point(x, y) for x, y in centroids]
    selection = np.array([sample_area.contains(point) for point in points])

    logger.info(
        f"Filtering {len(adata)} cells to {selection.sum()} cells within the sample area"
    )

    return adata[selection, :]


def filter_anndata_to_min_transcripts(adata: anndata.AnnData, min_transcripts: int):
    total_per_cell = np.array(adata.X.sum(axis=1)).squeeze()

    selection = total_per_cell >= min_transcripts

    logger.info(
        f"Filtering {len(adata)} cells to {selection.sum()} cells with more than {min_transcripts} transcripts"
    )

    return adata[selection, :]


def subset_anndata(adata: anndata.AnnData, n_cells: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_cells = min(n_cells, len(adata))
    selection = rng.choice(len(adata), n_cells, replace=False)
    return adata[selection, :]


def transform_shapefile_to_rasterized_space(
    gdf: geopandas.GeoDataFrame,
    resolution: float,
    sample_area: Optional[tuple[float, float, float, float]] = None,
):
    clipped = gdf.clip(sample_area)
    if sample_area is not None:
        clipped["geometry"] = clipped.geometry.translate(
            xoff=-sample_area[0], yoff=-sample_area[1]
        )
    clipped["geometry"] = clipped.geometry.scale(
        xfact=1 / resolution, yfact=1 / resolution, origin=(0, 0)
    )

    return clipped


def transform_bbox_to_slide_space(
    bbox: shapely.Polygon,
    resolution: float,
    sample_area: Optional[tuple[float, float, float, float]] = None,
):
    if sample_area is not None:
        bbox = translate(bbox, xoff=sample_area[0], yoff=sample_area[1])
    bbox = scale(
        bbox,
        xfact=resolution,
        yfact=resolution,
        origin=(sample_area[0], sample_area[1]),
    )

    return (
        math.floor(bbox.bounds[0]),
        math.floor(bbox.bounds[1]),
        math.ceil(bbox.bounds[2]),
        math.ceil(bbox.bounds[3]),
    )


def transform_bbox_to_raster_space(
    bbox: shapely.Polygon,
    resolution: float,
    sample_area: Optional[tuple[float, float, float, float]] = None,
):
    if sample_area is not None:
        bbox = translate(bbox, xoff=-sample_area[0], yoff=-sample_area[1])

    bbox = scale(bbox, xfact=resolution, yfact=resolution, origin=(0, 0))

    return (
        math.floor(bbox.bounds[0]),
        math.floor(bbox.bounds[1]),
        math.ceil(bbox.bounds[2]),
        math.ceil(bbox.bounds[3]),
    )


def transform_shapefile_to_slide_space(
    gdf: geopandas.GeoDataFrame,
    resolution: float,
    sample_area: Optional[tuple[float, float, float, float]] = None,
):
    gdf["geometry"] = gdf.geometry.translate(xoff=sample_area[0], yoff=sample_area[1])
    gdf["geometry"] = gdf.geometry.scale(
        xfact=resolution, yfact=resolution, origin=(sample_area[0], sample_area[1])
    )

    return gdf


def bbox_geometry_to_rasterized_slice(
    bbox: shapely.Polygon,
    resolution: float,
    sample_area: Optional[tuple[float, float, float, float]] = None,
) -> tuple[int, int, int, int]:
    if sample_area is not None:
        bbox = translate(bbox, xoff=-sample_area[0], yoff=-sample_area[1])
    bbox = scale(bbox, xfact=1 / resolution, yfact=1 / resolution, origin=(0, 0))

    return (
        math.floor(bbox.bounds[0]),
        math.floor(bbox.bounds[1]),
        math.ceil(bbox.bounds[2]),
        math.ceil(bbox.bounds[3]),
    )


def get_roi(resolution, labels, size=200):
    pixel_size = int(size / resolution)

    tiles = get_indexed_tiles(
        extent=labels.shape, tile_size=(pixel_size, pixel_size), overlap=0.0
    )

    n_nuclei = {}

    for tile_idx, bounds in tiles.items():
        x1, y1, x2, y2 = bounds
        tile = labels[y1:y2, x1:x2]
        n_nuclei[tile_idx] = len(np.unique(tile[tile > 0]))

    median_n_nuclei = sorted(list(n_nuclei.values()))[len(n_nuclei) // 2]

    for tile_idx, n in n_nuclei.items():
        if n == median_n_nuclei:
            x1, y1, x2, y2 = tiles[tile_idx]
            return x1, y1, x2, y2


def normalized_radians_to_radians(arr):
    return (arr * 2 * np.pi) - np.pi


def calculate_center_of_mass(tensor, x):
    """
    Calculate the center of mass for a 2D tensor where all values equal to x are considered.

    Parameters:
    tensor (numpy.ndarray): 2D tensor of integer values
    x (int): Value to consider for center of mass calculation

    Returns:
    tuple: (row_center, col_center) representing the center of mass coordinates
    """
    # Create a mask where values equal x
    mask = tensor == x

    # If no values equal x, return None
    if not np.any(mask):
        return None

    # Get coordinates of all points where value equals x
    points = np.argwhere(mask)

    # Calculate center of mass (mean of all coordinates)
    center_of_mass = points.mean(axis=0)

    # Return as (row, column) tuple
    return (center_of_mass[0] + 0.5, center_of_mass[1] + 0.5)


def reassign_angles_for_centroids(labels: torch.tensor, centroids: torch.tensor):
    """
    :param labels: H x W array of labels
    :param centroids: dictionary of centroids
    :return: H x W array of angles
    """
    angles = torch.zeros_like(labels).float()

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] <= 0:
                continue
            cell_idx = labels[x, y] - 1
            centroid = centroids[cell_idx.item()]
            x_component = centroid[0] - (x + 0.5)
            y_component = centroid[1] - (y + 0.5)
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] += (angle[1] + torch.pi) / (2 * torch.pi)
    return angles


def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def raster_to_polygon(raster):
    # get coordinates of true values in nparray
    x1, y1 = np.where(raster)
    x2 = x1 + 1
    y2 = y1 + 1

    shapes = []
    for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
        shapes.append(box(x1, y1, x2, y2))

    return shapely.union_all(shapes)


def cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def randomly_bisect_and_mask_segments(labels, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    output = np.zeros_like(labels)

    pixel_geo_df = create_pixel_geodf(labels.shape[0], labels.shape[1], resolution=1)

    for i in range(0, labels.max()):
        mask = labels == i + 1
        polygon = raster_to_polygon(mask)

        if polygon.area == 0.0:
            continue

        # get bounding box of polygon
        x1, y1, x2, y2 = polygon.bounds
        bbox = box(x1, y1, x2, y2)

        centroid = polygon.centroid

        # sample a random angle and create a line through the centroid at that angle
        angle = rng.uniform(0, 2 * np.pi)
        line = shapely.geometry.LineString(
            [
                (centroid.x + 1000 * np.cos(angle), centroid.y + 1000 * np.sin(angle)),
                (centroid.x - 1000 * np.cos(angle), centroid.y - 1000 * np.sin(angle)),
            ]
        )

        # clip the line to the polygon
        clipped = line.intersection(bbox)

        # get length of line
        length = clipped.length

        # sample random point on line
        truncation = rng.uniform(0, length)

        # drop the whole nucleus if we cut at an extreme
        if (truncation / length) < 0.1:
            continue

        # or keep it all if we would cut at the other extreme
        if (truncation / length) > 0.9:
            output[mask] = i + 1
            continue

        point = clipped.interpolate(rng.uniform(0, length))

        # create a line perpendicular to the original line at the point
        angle = angle + np.pi / 2
        line = shapely.geometry.LineString(
            [
                (point.x + 1000 * np.cos(angle), point.y + 1000 * np.sin(angle)),
                (point.x - 1000 * np.cos(angle), point.y - 1000 * np.sin(angle)),
            ]
        )

        bisected_bbox = cut_polygon_by_line(bbox, line)

        # randomly choose one of the two polygons
        bisected_bbox_half = rng.choice(bisected_bbox)

        clipped_polygon = polygon.intersection(bisected_bbox_half)

        pixels_clipped = pixel_geo_df.clip(clipped_polygon)
        pixels_clipped = pixels_clipped[pixels_clipped.geometry.area > 0.2]
        output[pixels_clipped.x_index.values, pixels_clipped.y_index.values] = i + 1
    return output


def create_pixel_geodf(width, height, resolution=1):
    n_width_pixels = math.ceil(width / resolution)
    n_height_pixels = math.ceil(height / resolution)

    rows = []
    for i in range(n_width_pixels):
        for j in range(n_height_pixels):
            rows.append(
                {
                    "x_index": i,
                    "y_index": j,
                    "geometry": box(
                        i * resolution,
                        j * resolution,
                        (i + 1) * resolution,
                        (j + 1) * resolution,
                    ),
                    "pixel_center_x": i * resolution + resolution / 2,
                    "pixel_center_y": j * resolution + resolution / 2,
                }
            )

    df = pd.DataFrame(rows)
    return geopandas.GeoDataFrame(df, geometry="geometry")
