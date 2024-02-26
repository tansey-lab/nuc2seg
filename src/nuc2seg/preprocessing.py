import math

import geopandas
import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import KDTree
import pandas as pd

from nuc2seg.data import Nuc2SegDataset
from nuc2seg.xenium import get_bounding_box, logger
from nuc2seg.celltyping import estimate_cell_types
from kneed import KneeLocator


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


def get_best_k(aic_scores, bic_scores):
    best_k_aic = np.argmin(aic_scores)
    best_k_bic = np.argmin(bic_scores)

    if best_k_bic == best_k_aic:
        return best_k_aic
    else:
        logger.warning(
            f"The best k according to AIC and BIC do not match ({best_k_aic} vs {best_k_bic}). Using BIC eblow to determine k"
        )
        kneedle = KneeLocator(
            x=np.arange(len(bic_scores)),
            y=bic_scores,
            S=2,
            curve="convex",
            direction="decreasing",
        )
        best_k = kneedle.elbow

        logger.info(f"BIC elbow to chose k: {best_k}")

        return best_k


def create_rasterized_dataset(
    nuclei_geo_df: geopandas.GeoDataFrame,
    tx_geo_df: geopandas.GeoDataFrame,
    sample_area: shapely.Polygon,
    resolution=1,
    foreground_nucleus_distance=1,
    background_nucleus_distance=10,
    background_transcript_distance=4,
    background_pixel_transcripts=5,
    min_n_celltypes=2,
    max_n_celltypes=25,
):
    n_genes = tx_geo_df["gene_id"].max() + 1

    x_min, x_max, y_min, y_max = get_bounding_box(sample_area)
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

    # Calculate the nearest transcript neighbors
    logger.info("Calculating the nearest transcript neighbors")
    transcript_xy = np.array(
        [tx_geo_df["x_location"].values, tx_geo_df["y_location"].values]
    ).T
    kdtree = KDTree(transcript_xy)

    # Get the distance to the k'th nearest transcript
    pixels_xy = np.array([labels_geo_df["X"].values, labels_geo_df["Y"].values]).T
    labels_geo_df["transcript_distance"] = kdtree.query(
        pixels_xy, k=background_pixel_transcripts + 1
    )[0][:, -1]

    # Assign pixels roughly on top of nuclei to belong to that nuclei label
    pixel_labels = np.zeros(labels_geo_df.shape[0], dtype=int) - 1
    nucleus_pixels = labels_geo_df["nucleus_distance"] <= foreground_nucleus_distance
    pixel_labels[nucleus_pixels] = labels_geo_df["nucleus_label"][nucleus_pixels]

    # Assign pixels to the background if they are far from nuclei and not near a dense region of transcripts
    background_pixels = (
        labels_geo_df["nucleus_distance"] > background_nucleus_distance
    ) & (labels_geo_df["transcript_distance"] > background_transcript_distance)
    pixel_labels[background_pixels] = 0

    # Convert back over to the grid format
    labels = np.zeros((x_size, y_size), dtype=int)
    labels[labels_geo_df["X"] - x_min, labels_geo_df["Y"] - y_min] = pixel_labels

    # Create a nuclei x gene count matrix
    tx_nuclei_geo_df = gpd.sjoin_nearest(
        tx_geo_df, nuclei_geo_df, distance_col="nucleus_distance"
    )
    nuclei_count_geo_df = tx_nuclei_geo_df[
        tx_nuclei_geo_df["nucleus_distance"] <= foreground_nucleus_distance
    ]

    # I think we have enough memory to just store this as a dense array
    nuclei_count_matrix = np.zeros((nuclei_geo_df.shape[0] + 1, n_genes), dtype=int)
    np.add.at(
        nuclei_count_matrix,
        (
            nuclei_count_geo_df["nucleus_label"].values.astype(int),
            nuclei_count_geo_df["gene_id"].values.astype(int),
        ),
        1,
    )

    # Assume for simplicity that it's a homogeneous poisson process for transcripts.
    # Add up all the transcripts in each pixel.
    tx_count_grid = np.zeros((x_size, y_size), dtype=int)
    np.add.at(
        tx_count_grid,
        (
            tx_geo_df["x_location"].values.astype(int) - x_min,
            tx_geo_df["y_location"].values.astype(int) - y_min,
        ),
        1,
    )

    logger.info("Estimating cell types")
    # Estimate the cell types

    celltyping_result = estimate_cell_types(
        nuclei_count_matrix,
        min_components=min_n_celltypes,
        max_components=max_n_celltypes,
    )

    (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        relative_expression,
    ) = (
        celltyping_result.aic_scores,
        celltyping_result.bic_scores,
        celltyping_result.final_expression_profiles,
        celltyping_result.final_prior_probs,
        celltyping_result.final_cell_types,
        celltyping_result.relative_expression,
    )

    best_k = get_best_k(aic_scores, bic_scores)
    logger.info(f"Best k: {best_k + 2}")

    # Estimate the background rate
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

    # Estimate the density of each cell type
    cell_type_probs = final_cell_types[best_k]

    # Assign hard labels to nuclei
    cell_type_labels = np.argmax(cell_type_probs, axis=1) + 1
    pixel_types = np.copy(labels)
    nuclei_mask = labels > 0
    pixel_types[nuclei_mask] = cell_type_labels[labels[nuclei_mask]]

    # Calculate the angle at which each pixel faces to point at its nearest nucleus centroid.
    # Normalize it to be in [0,1]
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

    n_classes = cell_type_probs.shape[-1]

    logger.info("Creating dataset")
    X = tx_geo_df["x_location"].values.astype(int) - x_min
    Y = tx_geo_df["y_location"].values.astype(int) - y_min
    G = tx_geo_df["gene_id"].values.astype(int)
    ds = Nuc2SegDataset(
        labels=labels,
        angles=angles,
        classes=pixel_types,
        transcripts=np.array([X, Y, G]).T,
        bbox=np.array([x_min, y_min, x_max, y_max]),
        n_classes=n_classes,
        n_genes=n_genes,
        resolution=1.0,
    )

    return ds, {
        "aic_scores": aic_scores,
        "bic_scores": bic_scores,
        "final_expression_profiles": final_expression_profiles,
        "final_prior_probs": final_prior_probs,
        "final_cell_types": final_cell_types,
        "relative_expression": relative_expression,
    }
