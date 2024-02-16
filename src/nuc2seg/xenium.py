import logging
import os

import shapely
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely import Polygon
from shapely.geometry import box

logger = logging.getLogger(__name__)


def create_shapely_rectangle(x1, y1, x2, y2):
    return box(x1, y1, x2, y2)


def get_bounding_box(poly: shapely.Polygon):
    coords = list(poly.exterior.coords)

    # Find the extreme vertices
    leftmost = min(coords, key=lambda point: point[0])
    rightmost = max(coords, key=lambda point: point[0])
    topmost = max(coords, key=lambda point: point[1])
    bottommost = min(coords, key=lambda point: point[1])

    return leftmost[0], rightmost[0], bottommost[1], topmost[1]


def filter_gdf_to_inside_polygon(gdf, polygon):
    selection_vector = gdf.geometry.apply(lambda x: polygon.contains(x))
    logging.info(
        f"Filtering {len(gdf)} points to {selection_vector.sum()} inside polygon"
    )
    return gdf[selection_vector]


def read_boundaries_into_polygons(
    boundaries_file,
    cell_id_column="cell_id",
    x_column_name="vertex_x",
    y_column_name="vertex_y",
):
    boundaries = pd.read_parquet(boundaries_file)
    geo_df = gpd.GeoDataFrame(
        boundaries,
        geometry=gpd.points_from_xy(
            boundaries[x_column_name], boundaries[y_column_name]
        ),
    )
    polys = geo_df.groupby(cell_id_column)["geometry"].apply(
        lambda x: Polygon(x.tolist())
    )
    return gpd.GeoDataFrame(polys)


def read_transcripts_into_points(
    transcripts_file, x_column_name="x_location", y_column_name="y_location"
):
    transcripts = pd.read_parquet(transcripts_file)

    tx_geo_df = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(
            transcripts[x_column_name], transcripts[y_column_name]
        ),
    )
    return tx_geo_df


def load_nuclei(nuclei_file: str, sample_area: shapely.Polygon):
    nuclei_geo_df = read_boundaries_into_polygons(nuclei_file)

    original_n_nuclei = nuclei_geo_df.shape[0]

    nuclei_geo_df = filter_gdf_to_inside_polygon(nuclei_geo_df, sample_area)

    logger.info(
        f"{original_n_nuclei-nuclei_geo_df.shape[0]} nuclei filtered after bounding to {sample_area}"
    )

    if nuclei_geo_df.empty:
        raise ValueError("No nuclei found in the sample area")

    nuclei_geo_df["nucleus_label"] = np.arange(1, nuclei_geo_df.shape[0] + 1)
    nuclei_geo_df["nucleus_centroid"] = nuclei_geo_df["geometry"].centroid
    nuclei_geo_df["nucleus_centroid_x"] = nuclei_geo_df["geometry"].centroid.x
    nuclei_geo_df["nucleus_centroid_y"] = nuclei_geo_df["geometry"].centroid.y

    logger.info(f"Loaded {nuclei_geo_df.shape[0]} nuclei.")

    return nuclei_geo_df


def load_and_filter_transcripts(
    transcripts_file: str, sample_area: shapely.Polygon, min_qv=20.0
):
    transcripts_df = read_transcripts_into_points(transcripts_file)

    original_count = len(transcripts_df)

    transcripts_df = filter_gdf_to_inside_polygon(transcripts_df, sample_area)
    transcripts_df.drop(columns=["nucleus_distance"], inplace=True)

    count_after_bbox = len(transcripts_df)

    logger.info(
        f"{original_count-count_after_bbox} tx filtered after bounding to {sample_area}"
    )

    # Filter out controls and low quality transcripts
    transcripts_df = transcripts_df[
        (transcripts_df["qv"] >= min_qv)
        & (~transcripts_df["feature_name"].str.startswith("NegControlProbe_"))
        & (~transcripts_df["feature_name"].str.startswith("antisense_"))
        & (~transcripts_df["feature_name"].str.startswith("NegControlCodeword_"))
        & (~transcripts_df["feature_name"].str.startswith("BLANK_"))
    ]

    count_after_quality_filtering = len(transcripts_df)

    logger.info(
        f"{count_after_bbox-count_after_quality_filtering} tx filtered after quality filtering"
    )

    if transcripts_df.empty:
        raise ValueError("No transcripts found in the sample area")

    # Assign a unique integer ID to each gene
    gene_ids = transcripts_df["feature_name"].unique()
    n_genes = len(gene_ids)
    mapping = dict(zip(gene_ids, np.arange(len(gene_ids))))
    transcripts_df["gene_id"] = transcripts_df["feature_name"].apply(
        lambda x: mapping.get(x, 0)
    )

    logger.info(
        f"Loaded {count_after_quality_filtering} transcripts. {n_genes} unique genes."
    )

    return transcripts_df


def plot_distribution_of_cell_types(cell_type_probs):
    n_rows = cell_type_probs.shape[1] // 3 + int(cell_type_probs.shape[1] % 3 > 0)
    n_cols = 3
    fig, axarr = plt.subplots(n_rows, n_cols, sharex=True)
    for idx in range(cell_type_probs.shape[1]):
        i, j = idx // 3, idx % 3
        axarr[i, j].hist(
            cell_type_probs[np.argmax(cell_type_probs, axis=1) == idx, idx], bins=200
        )
    plt.show()