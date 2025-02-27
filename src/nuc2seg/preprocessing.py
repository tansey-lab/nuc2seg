import math
import os.path
import logging
from multiprocessing.pool import ThreadPool

import geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
from blended_tiling import TilingModule
from scipy.spatial import KDTree
from shapely import box
from nuc2seg.utils import transform_shapefile_to_rasterized_space

from nuc2seg.data import (
    RasterizedDataset,
    Nuc2SegDataset,
)
from nuc2seg.utils import generate_tiles

logger = logging.getLogger(__name__)


def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


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


def create_rasterized_dataset(
    prior_segmentation_gdf: geopandas.GeoDataFrame,
    tx_geo_df: geopandas.GeoDataFrame,
    sample_area: shapely.Polygon,
    resolution=1,
    foreground_distance=1,
    background_distance=10,
    background_transcript_distance=4,
    background_pixel_transcripts=5,
):

    prior_segmentation_gdf = prior_segmentation_gdf.copy()
    n_genes = tx_geo_df["gene_id"].max() + 1
    if "transcript_id" in tx_geo_df.columns:
        del tx_geo_df["transcript_id"]
    tx_geo_df.reset_index(names="transcript_id", inplace=True)

    x_min, y_min, x_max, y_max = sample_area.bounds

    width = x_max - x_min
    height = y_max - y_min

    tx_geo_df = transform_shapefile_to_rasterized_space(
        tx_geo_df, resolution, sample_area.bounds
    )

    prior_segmentation_gdf = transform_shapefile_to_rasterized_space(
        prior_segmentation_gdf, resolution, sample_area.bounds
    )
    if "segment_id" in prior_segmentation_gdf.columns:
        del prior_segmentation_gdf["segment_id"]
    prior_segmentation_gdf.reset_index(names="segment_id", inplace=True)
    prior_segmentation_gdf["segment_id"] += 1

    logger.info("Creating pixel geometry dataframe")
    # Create a dataframe with an entry for every pixel
    idx_geo_df = create_pixel_geodf(width, height, resolution)

    x_max_idx = idx_geo_df["x_index"].max()
    y_max_idx = idx_geo_df["y_index"].max()

    logger.info("Find the nearest segment to each pixel")
    # Find the nearest segment to each pixel
    prior_segmentation_gdf["centroid_x"] = prior_segmentation_gdf.centroid.x
    prior_segmentation_gdf["centroid_y"] = prior_segmentation_gdf.centroid.y
    labels_geo_df = gpd.sjoin_nearest(
        idx_geo_df,
        prior_segmentation_gdf,
        how="left",
        distance_col="distance",
        max_distance=background_distance,
    )
    labels_geo_df.rename(columns={"index_right": "prior_segment_id"}, inplace=True)
    # break ties arbitrarily
    labels_geo_df.drop_duplicates(subset=["x_index", "y_index"], inplace=True)
    logger.info("Calculating the nearest transcript neighbors")
    transcript_xy = np.array(
        [
            np.floor(tx_geo_df.centroid.x.values).astype(int),
            np.floor(tx_geo_df.centroid.y.values).astype(int),
        ]
    ).T
    kdtree = KDTree(transcript_xy)

    logger.info("Get the distance to the k'th nearest transcript")
    pixels_xy = np.array(
        [labels_geo_df["x_index"].values, labels_geo_df["y_index"].values]
    ).T
    labels_geo_df["distance_to_farthest_transcript"] = kdtree.query(
        pixels_xy, k=background_pixel_transcripts
    )[0][:, -1]

    labels_geo_df["in_transcript_dense_area"] = labels_geo_df[
        "distance_to_farthest_transcript"
    ] < (background_transcript_distance / resolution)

    logger.info("Assign pixels roughly on top of nuclei to belong to that nuclei label")
    pixel_labels = np.zeros(labels_geo_df.shape[0], dtype=int) - 1
    segmented_pixels = labels_geo_df["distance"] <= foreground_distance
    pixel_labels[segmented_pixels] = (
        labels_geo_df["prior_segment_id"][segmented_pixels] + 1
    )

    logger.info(
        "Assign pixels to the background if they are far from nuclei and not near a dense region of transcripts"
    )
    labels_geo_df["is_background_pixel"] = (
        (labels_geo_df["distance"] > background_distance)
        | labels_geo_df["distance"].isna()
    ) & (~labels_geo_df["in_transcript_dense_area"])
    pixel_labels[labels_geo_df["is_background_pixel"].values] = 0

    logger.info("Convert pixel labels to a grid")
    labels = pixel_labels.reshape(x_max_idx + 1, y_max_idx + 1)

    # Assume for simplicity that it's a homogeneous poisson process for transcripts.
    # Add up all the transcripts in each pixel.
    logger.info("Add up all transcripts in each pixel")
    tx_geo_df = tx_geo_df.sjoin(idx_geo_df, how="inner")

    # Arbtirarily drop ties (transcript is on pixel edge)
    tx_geo_df.drop_duplicates(subset=["transcript_id"], inplace=True)

    tx_count_grid = np.zeros((x_max_idx + 1, y_max_idx + 1), dtype=int)
    np.add.at(
        tx_count_grid,
        (tx_geo_df["x_index"], tx_geo_df["y_index"]),
        1,
    )

    # Estimate the background rate
    logger.info("Estimating background rate")
    tx_background_mask = labels[tx_geo_df["x_index"], tx_geo_df["y_index"]] == 0
    background_probs = np.zeros(n_genes)
    tx_geo_df_background = tx_geo_df[tx_background_mask]
    for g in range(n_genes):
        background_probs[g] = (tx_geo_df_background["gene_id"] == g).sum() + 1

    # Calculate the angle at which each pixel faces to point at its nearest nucleus centroid.
    # Normalize it to be in [0,1]
    logger.info(
        "Calculating the angle at which each pixel faces to point at its nearest nucleus centroid"
    )
    labels_geo_df["angle"] = (
        cart2pol(
            labels_geo_df["centroid_x"].values - labels_geo_df["pixel_center_x"].values,
            labels_geo_df["centroid_y"].values - labels_geo_df["pixel_center_y"].values,
        )[1]
        + np.pi
    ) / (2 * np.pi)
    angles = np.zeros(labels.shape)
    angles[labels_geo_df["x_index"], labels_geo_df["y_index"]] = labels_geo_df[
        "angle"
    ].values

    logger.info("Creating dataset")
    transcripts_arr = np.column_stack(
        [
            tx_geo_df["x_index"].values,
            tx_geo_df["y_index"].values,
            (tx_geo_df["gene_id"].values.astype(int)),
        ]
    )

    logger.info("Created transcripts array")

    ds = RasterizedDataset(
        labels=labels,
        angles=angles,
        transcripts=transcripts_arr,
        bbox=np.array([x_min, y_min, x_max, y_max]),
        n_genes=n_genes,
        resolution=resolution,
    )

    return ds


def tile_transcripts_to_disk(
    transcripts, bounds, tile_size, overlap, output_dir, output_format: str, n_threads=8
):
    if bounds is None:
        x_offset = 0
        y_offset = 0
    else:
        x_offset = bounds[0]
        y_offset = bounds[1]

    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=(bounds[2] - bounds[0], bounds[3] - bounds[1]),
    )

    total_n_transcripts = len(transcripts)
    thread_pool = ThreadPool(n_threads)
    pbar = tqdm.tqdm(total=total_n_transcripts)

    def write_tile(tpl):
        (idx, x1, y1, x2, y2) = tpl
        selection = (
            (transcripts["x_location"] >= (x1 + x_offset))
            & (transcripts["x_location"] < (x2 + x_offset))
            & (transcripts["y_location"] >= (y1 + y_offset))
            & (transcripts["y_location"] < (y2 + y_offset))
        )

        transcripts_view = transcripts[selection]

        output_path = os.path.join(output_dir, f"transcript_tile_{idx}.{output_format}")
        if output_format == "csv":
            transcripts_view.to_csv(output_path, index=False)
        elif output_format == "parquet":
            transcripts_view.to_parquet(output_path, index=False)
        pbar.update(len(transcripts_view))

    index_generator = [
        (idx, x1, y1, x2, y2)
        for idx, (x1, y1, x2, y2) in enumerate(
            generate_tiles(
                tiler,
                x_extent=(bounds[2] - bounds[0]),
                y_extent=(bounds[3] - bounds[1]),
                overlap_fraction=overlap,
                tile_size=tile_size,
            )
        )
    ]

    async_result = thread_pool.map_async(write_tile, index_generator)
    async_result.get()
    thread_pool.close()
    thread_pool.join()


def tile_nuclei_to_disk(
    nuclei_df, bounds, tile_size, overlap, output_dir, output_format: str, n_threads=8
):
    if bounds is None:
        x_offset = 0
        y_offset = 0
    else:
        x_offset = bounds[0]
        y_offset = bounds[1]

    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=(bounds[2] - bounds[0], bounds[3] - bounds[1]),
    )

    total_n_nuclei = len(nuclei_df)
    pbar = tqdm.tqdm(total=total_n_nuclei)
    thread_pool = ThreadPool(n_threads)

    def write_tile(tpl):
        (idx, x1, y1, x2, y2) = tpl
        selection = (
            (nuclei_df["vertex_x"] >= (x1 + x_offset))
            & (nuclei_df["vertex_x"] < (x2 + x_offset))
            & (nuclei_df["vertex_y"] >= (y1 + y_offset))
            & (nuclei_df["vertex_y"] < (y2 + y_offset))
        )
        filtered_df = nuclei_df[selection]

        unique_nuclei_ids = filtered_df["cell_id"].unique()

        filtered_df = filtered_df[filtered_df["cell_id"].isin(unique_nuclei_ids)]

        output_path = os.path.join(output_dir, f"nuclei_tile_{idx}.{output_format}")
        if output_format == "csv":
            filtered_df.to_csv(output_path, index=False)
        elif output_format == "parquet":
            filtered_df.to_parquet(output_path, index=False)
        pbar.update(len(filtered_df))

    index_generator = [
        (idx, x1, y1, x2, y2)
        for idx, (x1, y1, x2, y2) in enumerate(
            generate_tiles(
                tiler,
                x_extent=(bounds[2] - bounds[0]),
                y_extent=(bounds[3] - bounds[1]),
                overlap_fraction=overlap,
                tile_size=tile_size,
            )
        )
    ]

    async_result = thread_pool.map_async(write_tile, index_generator)
    async_result.get()
    thread_pool.close()
    thread_pool.join()


def tile_dataset_to_disk(
    dataset: Nuc2SegDataset, tile_size, overlap, output_dir, n_threads=8
):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=(dataset.x_extent_pixels, dataset.y_extent_pixels),
    )

    thread_pool = ThreadPool(n_threads)

    pbar = tqdm.tqdm(total=tiler.num_tiles(), desc="Tiling dataset")

    def write_tile(tpl):
        (idx, x1, y1, x2, y2) = tpl
        dataset_tile = dataset.clip((x1, y1, x2, y2))
        output_path = os.path.join(output_dir, f"dataset_tile_{idx}.h5")
        dataset_tile.save_h5(output_path, compression=None)
        pbar.update(1)

    index_generator = [
        (idx, x1, y1, x2, y2)
        for idx, (x1, y1, x2, y2) in enumerate(
            generate_tiles(
                tiler,
                x_extent=dataset.x_extent_pixels,
                y_extent=dataset.y_extent_pixels,
                overlap_fraction=overlap,
                tile_size=tile_size,
            )
        )
    ]

    async_result = thread_pool.map_async(write_tile, index_generator)
    async_result.get()
    thread_pool.close()
    thread_pool.join()
