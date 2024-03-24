import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
import json
import logging
import anndata

from scipy.sparse import csr_matrix
from nuc2seg.data import generate_tiles
from blended_tiling import TilingModule
from shapely import box

logger = logging.getLogger(__name__)


def filter_gdf_to_tile_boundary(
    gdf: gpd.GeoDataFrame, tile_idx: int, tile_size, base_size, overlap
):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=base_size,
    )
    tile_masks = tiler.get_tile_masks()[:, 0, :, :]
    bboxes = generate_tiles(
        tiler,
        x_extent=base_size[0],
        y_extent=base_size[1],
        tile_size=tile_size,
        overlap_fraction=overlap,
    )

    masks_and_bboxes = list(zip(tile_masks, bboxes))

    mask = masks_and_bboxes[tile_idx][0].detach().cpu().numpy()
    bbox = masks_and_bboxes[tile_idx][1]

    mask = (mask > 0.5).astype(bool)
    x, y = np.where(mask)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    offset_x = bbox[0]
    offset_y = bbox[1]

    selection_box = box(
        x_min + offset_x,
        y_min + offset_y,
        x_max + offset_x + 1,
        y_max + offset_y + 1,
    )

    gdf["intersection_area"] = gdf.geometry.apply(
        lambda g: g.intersection(selection_box).area
    )

    # Step 2: Calculate the percentage of the intersection area relative to the polygon's own area
    gdf["intersection_percentage"] = gdf["intersection_area"] / gdf.geometry.area

    return gdf[gdf["intersection_percentage"] > 0.5]


def stitch_shapes(shapes: list[gpd.GeoDataFrame], tile_size, base_size, overlap):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=base_size,
    )

    bboxes = generate_tiles(
        tiler,
        x_extent=base_size[0],
        y_extent=base_size[1],
        tile_size=tile_size,
        overlap_fraction=overlap,
    )

    centroids = []

    for idx, bbox in enumerate(bboxes):
        centroids.append(
            {
                "tile_idx": idx,
                "geometry": shapely.Point(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
            }
        )
    centroid_gdf = gpd.GeoDataFrame(centroids, geometry="geometry")

    results = []

    for tile_idx, shapefile in enumerate(shapes):
        joined_to_centroids = gpd.sjoin_nearest(
            shapefile,
            centroid_gdf,
        )
        # dedupe joined_to_centroids
        joined_to_centroids = joined_to_centroids.drop_duplicates(subset=["cell"])

        filtered_shapes = joined_to_centroids[
            joined_to_centroids["tile_idx"] == tile_idx
        ]

        results.append(filtered_shapes)

    result_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

    if "index_right" in result_gdf:
        del result_gdf["index_right"]
    if "index_left" in result_gdf:
        del result_gdf["index_left"]
    return result_gdf


def read_baysor_shapefile(shapes_fn):
    with open(shapes_fn) as f:
        geojson_data = json.load(f)

    records = []
    for geometry in geojson_data["geometries"]:
        if len(geometry["coordinates"][0]) <= 3:
            logger.debug(
                f"Skipping cell with {len(geometry['coordinates'][0])} vertices"
            )
            continue
        polygon = shapely.Polygon(geometry["coordinates"][0])
        records.append({"geometry": polygon, "cell": geometry["cell"]})

    return gpd.GeoDataFrame(records, geometry="geometry")


def read_baysor_shapes_with_cluster_assignment(
    shapes_fn, transcripts_fn, x_column_name="x", y_column_name="y"
) -> gpd.GeoDataFrame:
    with open(shapes_fn) as f:
        geojson_data = json.load(f)

    records = []
    for geometry in tqdm.tqdm(geojson_data["geometries"]):
        if len(geometry["coordinates"][0]) <= 3:
            logger.debug(
                f"Skipping cell with {len(geometry['coordinates'][0])} vertices"
            )
            continue
        polygon = shapely.Polygon(geometry["coordinates"][0])
        records.append({"geometry": polygon, "cell": geometry["cell"]})

    gdf = gpd.GeoDataFrame(records)

    transcripts_df = pd.read_csv(
        transcripts_fn,
        usecols=["cell", "cluster", "gene", "assignment_confidence", "x", "y"],
    )
    tx_geo_df = gpd.GeoDataFrame(
        transcripts_df,
        geometry=gpd.points_from_xy(
            transcripts_df[x_column_name], transcripts_df[y_column_name]
        ),
    )

    transcripts_df["cell_id"] = transcripts_df["cell"].apply(
        lambda x: int(x.split("-")[-1])
    )
    cell_to_cluster = transcripts_df[["cell_id", "cluster"]].drop_duplicates()

    result = gdf.merge(cell_to_cluster, left_on="cell", right_on="cell_id")
    del result["cell_id"]
    return result, tx_geo_df
