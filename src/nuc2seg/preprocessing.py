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


def create_pixel_geodf(x_min, x_max, y_min, y_max):
    # Create the list of all pixels
    grid_df = pd.DataFrame(
        np.array(
            np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        ).T.reshape(-1, 2),
        columns=["X", "Y"],
    )

    # Convert the xy locations to a geopandas data frame
    idx_geo_df = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.points_from_xy(grid_df["X"], grid_df["Y"]),
    )

    return idx_geo_df


def create_rasterized_dataset(
    nuclei_geo_df: geopandas.GeoDataFrame,
    tx_geo_df: geopandas.GeoDataFrame,
    sample_area: shapely.Polygon,
    resolution=1,
    foreground_nucleus_distance=1,
    background_nucleus_distance=10,
    background_transcript_distance=4,
    background_pixel_transcripts=5,
):
    n_genes = tx_geo_df["gene_id"].max() + 1

    x_min, y_min, x_max, y_max = sample_area.bounds
    x_min, x_max = math.floor(x_min), math.ceil(x_max)
    y_min, y_max = math.floor(y_min), math.ceil(y_max)

    x_size = (x_max - x_min) + 1
    y_size = (y_max - y_min) + 1

    logger.info("Creating pixel geometry dataframe")
    # Create a dataframe with an entry for every pixel
    idx_geo_df = create_pixel_geodf(x_max=x_max, x_min=x_min, y_min=y_min, y_max=y_max)

    logger.info("Find the nearest nucleus to each pixel")
    # Find the nearest nucleus to each pixel
    labels_geo_df = gpd.sjoin_nearest(
        idx_geo_df, nuclei_geo_df, how="left", distance_col="nucleus_distance"
    )
    labels_geo_df.rename(columns={"index_right": "nucleus_id_xenium"}, inplace=True)

    logger.info("Calculating the nearest transcript neighbors")
    transcript_xy = np.array(
        [tx_geo_df["x_location"].values, tx_geo_df["y_location"].values]
    ).T
    kdtree = KDTree(transcript_xy)

    logger.info("Get the distance to the k'th nearest transcript")
    pixels_xy = np.array([labels_geo_df["X"].values, labels_geo_df["Y"].values]).T
    labels_geo_df["transcript_distance"] = kdtree.query(
        pixels_xy, k=background_pixel_transcripts + 1
    )[0][:, -1]

    logger.info("Assign pixels roughly on top of nuclei to belong to that nuclei label")
    pixel_labels = np.zeros(labels_geo_df.shape[0], dtype=int) - 1
    nucleus_pixels = labels_geo_df["nucleus_distance"] <= foreground_nucleus_distance
    pixel_labels[nucleus_pixels] = labels_geo_df["nucleus_label"][nucleus_pixels]

    logger.info(
        "Assign pixels to the background if they are far from nuclei and not near a dense region of transcripts"
    )
    background_pixels = (
        labels_geo_df["nucleus_distance"] > background_nucleus_distance
    ) & (labels_geo_df["transcript_distance"] > background_transcript_distance)
    pixel_labels[background_pixels] = 0

    logger.info("Convert pixel labels to a grid")
    labels = np.zeros((x_size, y_size), dtype=int)
    labels[labels_geo_df["X"] - x_min, labels_geo_df["Y"] - y_min] = pixel_labels

    # Assume for simplicity that it's a homogeneous poisson process for transcripts.
    # Add up all the transcripts in each pixel.
    logger.info("Add up all transcripts in each pixel")
    tx_count_grid = np.zeros((x_size, y_size), dtype=int)
    np.add.at(
        tx_count_grid,
        (
            tx_geo_df["x_location"].values.astype(int) - x_min,
            tx_geo_df["y_location"].values.astype(int) - y_min,
        ),
        1,
    )

    # Estimate the background rate
    logger.info("Estimating background rate")
    tx_background_mask = (
        labels[
            tx_geo_df["x_location"].values.astype(int) - x_min,
            tx_geo_df["y_location"].values.astype(int) - y_min,
        ]
        == 0
    )
    background_probs = np.zeros(n_genes)
    tx_geo_df_background = tx_geo_df[tx_background_mask]
    for g in range(n_genes):
        background_probs[g] = (tx_geo_df_background["gene_id"] == g).sum() + 1

    # Calculate the angle at which each pixel faces to point at its nearest nucleus centroid.
    # Normalize it to be in [0,1]
    logger.info(
        "Calculating the angle at which each pixel faces to point at its nearest nucleus centroid"
    )
    labels_geo_df["nucleus_angle"] = (
        cart2pol(
            labels_geo_df["nucleus_centroid_x"].values - labels_geo_df["X"].values,
            labels_geo_df["nucleus_centroid_y"].values - labels_geo_df["Y"].values,
        )[1]
        + np.pi
    ) / (2 * np.pi)
    angles = np.zeros(labels.shape)
    angles[labels_geo_df["X"].values - x_min, labels_geo_df["Y"].values - y_min] = (
        labels_geo_df["nucleus_angle"].values
    )

    logger.info("Creating dataset")
    transcripts_arr = np.column_stack(
        [
            (tx_geo_df["x_location"].values.astype(int) - x_min),
            (tx_geo_df["y_location"].values.astype(int) - y_min),
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
        resolution=1.0,
    )

    return ds


def create_nuc2seg_dataset(
    rasterized_dataset: RasterizedDataset,
    cell_type_probs: np.array,
):
    n_classes = cell_type_probs.shape[1]

    # Assign hard labels to nuclei
    cell_type_labels = np.argmax(cell_type_probs, axis=1) + 1
    pixel_types = np.copy(rasterized_dataset.labels)
    nuclei_mask = rasterized_dataset.labels > 0
    pixel_types[nuclei_mask] = cell_type_labels[
        rasterized_dataset.labels[nuclei_mask] - 1
    ]

    ds = Nuc2SegDataset(
        labels=rasterized_dataset.labels,
        angles=rasterized_dataset.angles,
        classes=pixel_types,
        transcripts=rasterized_dataset.transcripts,
        bbox=rasterized_dataset.bbox,
        n_classes=n_classes,
        n_genes=rasterized_dataset.n_genes,
        resolution=rasterized_dataset.resolution,
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
