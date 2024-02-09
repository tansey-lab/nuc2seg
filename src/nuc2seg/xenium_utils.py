import argparse
import logging
import math
import os
import sys

import geopandas
import shapely
import tqdm
from scipy.special import softmax
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely import Polygon
from scipy.special import logsumexp
from scipy.stats import poisson
from scipy.spatial import KDTree
from shapely.geometry import box
import h5py

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


def spatial_join_polygons_and_transcripts(
    boundaries: gpd.GeoDataFrame, transcripts: gpd.GeoDataFrame
):
    joined_gdf = gpd.sjoin(transcripts, boundaries, how="left")

    unique_vals = joined_gdf["index_right"][joined_gdf["index_right"].notna()].unique()

    mapping = dict(zip(unique_vals, np.arange(len(unique_vals))))

    joined_gdf["nucleus_id"] = joined_gdf["index_right"].apply(
        lambda x: mapping.get(x, 0)
    )

    return joined_gdf


def plot_spatial_join(joined_gdf, polygon_gdf, output_dir, cell_id_column="nucleus_id"):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=1000)
    polygon_gdf.plot(ax=ax, edgecolor="black")
    joined_gdf[joined_gdf["nucleus_id"] == 0].plot(ax=ax, color="black", markersize=1)
    joined_gdf[joined_gdf["nucleus_id"] > 0].plot(
        ax=ax,
        column=cell_id_column,
        categorical=True,
        legend=False,
        markersize=1,
        cmap="rainbow",
    )

    plt.savefig(os.path.join(output_dir, "transcript_and_boundary.pdf"))


## TODO: Maybe replace this with a library that does the same thing
def estimate_cell_types(
    gene_counts,
    min_components=2,
    max_components=25,
    max_em_steps=100,
    tol=1e-4,
    warm_start=False,
):

    n_nuclei, n_genes = gene_counts.shape

    # Randomly initialize cell type profiles
    cur_expression_profiles = np.random.dirichlet(np.ones(n_genes), size=min_components)

    # Initialize probabilities to be uniform
    cur_prior_probs = np.ones(min_components) / min_components

    # No need to initialize cell type assignments
    cur_cell_types = None

    # Track BIC and AIC scores
    aic_scores = np.zeros(max_components - min_components + 1)
    bic_scores = np.zeros(max_components - min_components + 1)

    # Iterate through every possible number of components
    (
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        final_expression_rates,
    ) = ([], [], [], [])
    prev_expression_profiles = np.zeros_like(cur_expression_profiles)
    for idx, n_components in enumerate(
        tqdm.trange(
            min_components, max_components + 1, desc="estimate_cell_types", position=0
        )
    ):

        # Warm start from the previous iteration
        if warm_start and n_components > min_components:
            # Expand and copy over the current parameters
            next_prior_probs = np.zeros(n_components)
            next_expression_profiles = np.zeros((n_components, n_genes))
            next_prior_probs[: n_components - 1] = cur_prior_probs
            next_expression_profiles[: n_components - 1] = cur_expression_profiles

            # Split the dominant cluster
            dominant = np.argmax(cur_prior_probs)
            split_prob = np.random.random()
            next_prior_probs[-1] = next_prior_probs[dominant] * split_prob
            next_prior_probs[dominant] = next_prior_probs[dominant] * (1 - split_prob)
            next_expression_profiles[-1] = cur_expression_profiles[
                dominant
            ] * split_prob + (1 - split_prob) * np.random.dirichlet(np.ones(n_genes))

            cur_prior_probs = next_prior_probs
            cur_expression_profiles = next_expression_profiles

            logger.debug("priors:", cur_prior_probs)
            logger.debug("expression:", cur_expression_profiles)
        else:
            # Cold start from a random location
            cur_expression_profiles = np.random.dirichlet(
                np.ones(n_genes), size=n_components
            )
            cur_prior_probs = np.ones(n_components) / n_components

        converge = tol + 1
        for step in tqdm.trange(
            max_em_steps,
            desc=f"EM for n_components {n_components}",
            unit="step",
            position=1,
        ):

            # E-step: estimate cell type assignments (posterior probabilities)
            logits = np.log(cur_prior_probs[None]) + (
                gene_counts[:, None] * np.log(cur_expression_profiles[None])
            ).sum(axis=2)
            cur_cell_types = softmax(logits, axis=1)

            # M-step (part 1): estimate cell type profiles
            prev_expression_profiles = np.array(cur_expression_profiles)
            cur_expression_profiles = (
                cur_cell_types[..., None] * gene_counts[:, None]
            ).sum(axis=0) + 1
            cur_expression_profiles = (
                cur_expression_profiles / (cur_cell_types.sum(axis=0) + 1)[:, None]
            )

            # M-step (part 2): estimate cell type probabilities
            cur_prior_probs = (cur_cell_types.sum(axis=0) + 1) / (
                cur_cell_types.shape[0] + n_components
            )
            logger.debug(f"cur_prior_probs: {cur_prior_probs}")

            # Track convergence of the cell type profiles
            converge = np.linalg.norm(
                prev_expression_profiles - cur_expression_profiles
            ) / np.linalg.norm(prev_expression_profiles)

            logger.debug(f"Convergence: {converge:.4f}")
            if converge <= tol:
                logger.debug(f"Stopping early.")
                break

        # Save the results
        final_expression_profiles.append(cur_expression_profiles)
        final_cell_types.append(cur_cell_types)
        final_prior_probs.append(cur_prior_probs)

        # Calculate BIC and AIC
        aic_scores[idx], bic_scores[idx] = aic_bic(
            gene_counts, cur_expression_profiles, cur_prior_probs
        )

        logger.debug(f"K={n_components}")
        logger.debug(f"AIC: {aic_scores[idx]:.4f}")
        logger.debug(f"BIC: {bic_scores[idx]:.4f}")

    return {
        "bic": bic_scores,
        "aic": aic_scores,
        "expression_profiles": final_expression_profiles,
        "prior_probs": final_prior_probs,
        "cell_types": final_cell_types,
    }


def aic_bic(gene_counts, expression_profiles, prior_probs):

    n_components = expression_profiles.shape[0]
    n_genes = expression_profiles.shape[1]
    n_samples = gene_counts.shape[0]

    dof = n_components * (n_genes - 1) + n_components - 1
    log_likelihood = logsumexp(
        np.log(prior_probs[None])
        + (gene_counts[:, None] * np.log(expression_profiles[None])).sum(axis=2),
        axis=1,
    ).sum()
    aic = -2 * log_likelihood + 2 * dof
    bic = -2 * log_likelihood + dof * np.log(n_samples)

    return aic, bic


"""
nuclei_file = 'data/nucleus_boundaries.parquet'
transcripts_file = 'data/transcripts.csv'
out_dir = 'data/'
pixel_stride=1
min_qv=20.0
foreground_nucleus_distance=1
background_nucleus_distance=10
background_transcript_distance=4
background_pixel_transcripts=5
tile_height=64
tile_width=64
"""


def calculate_pixel_loglikes(
    tx_geo_df, tx_count_grid, expression_profiles, expression_rates, x_min, y_min
):
    # Create zero-padded arrays of pixels x transcript counts and IDs
    # NOTE: this is a painfully slow way of doing this. it's just a quick and dirty solution.
    max_transcript_count = tx_count_grid.max() + 1
    transcript_ids = np.zeros(tx_count_grid.shape + (max_transcript_count,), dtype=int)
    transcript_counts = np.zeros(
        tx_count_grid.shape + (max_transcript_count,), dtype=int
    )
    for row_idx, row in tqdm.tqdm(
        tx_geo_df.iterrows(), desc="calculate_pixel_loglikes"
    ):
        x, y = (
            math.floor(row["x_location"]) - x_min,
            math.floor(row["y_location"]) - y_min,
        )
        tx_mask = transcript_ids[x, y] == row["gene_id"]
        if tx_mask.sum() != 0:
            gene_idx = np.argmax(tx_mask)
            transcript_counts[x, y, gene_idx] += 1
        else:
            tx_mask = transcript_counts[x, y] == 0
            gene_idx = np.argmax(tx_mask)
            transcript_ids[x, y, gene_idx] += row["gene_id"]
            transcript_counts[x, y, gene_idx] += 1

    # Track how many transcripts are in each pixel and how much padding we need
    totals = transcript_counts.sum(axis=-1)
    max_count = totals.max()

    # Calculate the log-likelihoods for each pixel having that many transcripts.
    # Be clever about this to avoid recalculating lots of duplicate PMF values.
    rate_loglike_uniques = poisson.logpmf(
        np.arange(max_count + 1)[:, None], expression_rates[None]
    )
    rate_loglikes = rate_loglike_uniques[totals.reshape(-1)].reshape(
        totals.shape + (expression_rates.shape[0],)
    )

    # Calculate the log-likelihoods for each pixel generating a certain set of transcripts.
    # NOTE: yes we could vectorize this, but that leads to memory issues.
    expression_loglikes = np.zeros_like(rate_loglikes)
    log_expression = np.log(expression_profiles)
    total = expression_loglikes.shape[0] * expression_loglikes.shape[1]
    pbar = tqdm.tqdm(total=total, desc="calculate_pixel_loglikes")

    for i in range(expression_loglikes.shape[0]):
        for j in range(expression_loglikes.shape[1]):
            expression_loglikes[i, j] = (
                transcript_counts[i, j, :, None]
                * log_expression.T[transcript_ids[i, j]]
            ).sum(axis=0)
            pbar.update(1)

    # Add the two log-likelihoods together to get the pixelwise log-likelihood
    return rate_loglikes, expression_loglikes


def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


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


def spatial_as_sparse_arrays(
    nuclei_geo_df: geopandas.GeoDataFrame,
    tx_geo_df: geopandas.GeoDataFrame,
    sample_area: shapely.Polygon,
    outdir: str,
    pixel_stride=1,
    foreground_nucleus_distance=1,
    background_nucleus_distance=10,
    background_transcript_distance=4,
    background_pixel_transcripts=5,
    tile_height=64,
    tile_width=64,
    tile_stride=48,
):
    """Creates a list of sparse CSC arrays. First array is the nuclei mask.
    All other arrays are the transcripts."""

    gene_ids = tx_geo_df["gene_id"].unique()

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
    results = estimate_cell_types(nuclei_count_matrix)
    best_k = 12

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
    cell_type_probs = results["cell_types"][best_k]

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

    # Create tiled images and labels from the giant image
    image_id = 0
    tile_locations = []
    tx_local_counts = []
    class_local_counts = []
    n_classes = cell_type_probs.shape[-1]
    if not os.path.exists(f"{outdir}/tiles/transcripts/"):
        os.makedirs(f"{outdir}/tiles/transcripts/")
    if not os.path.exists(f"{outdir}/tiles/labels/"):
        os.makedirs(f"{outdir}/tiles/labels/")
    if not os.path.exists(f"{outdir}/tiles/angles/"):
        os.makedirs(f"{outdir}/tiles/angles/")
    if not os.path.exists(f"{outdir}/tiles/classes/"):
        os.makedirs(f"{outdir}/tiles/classes/")

    n_tiles_total = (
        np.arange(y_max, x_max + 1, tile_stride).shape[0]
        * np.arange(y_min, y_max + 1, tile_stride).shape[0]
    )

    progress_bar = tqdm.tqdm(total=n_tiles_total, desc="Processing tiles")

    logger.info("Creating tiles")

    with h5py.File(f"{outdir}/data.h5", "w") as f:
        f.create_dataset("nuclei", data=labels, compression="gzip")
        f.create_dataset("angles", data=angles, compression="gzip")
        f.create_dataset("classes", data=pixel_types, compression="gzip")
        X = tx_geo_df["x_location"].values.astype(int) - x_min
        Y = tx_geo_df["y_location"].values.astype(int) - y_min
        G = tx_geo_df["gene_id"].values.astype(int)
        f.create_dataset("transcripts", data=np.array([X, Y, G]).T, compression="gzip")
        f.create_dataset("bbox", data=np.array([x_min, x_max, y_min, y_max]))
        f.attrs["n_classes"] = n_classes
        f.attrs["n_genes"] = n_genes

    return

    for x_start in np.arange(y_max, x_max + 1, tile_stride):
        # Handle edge cases
        x_start = min(x_start, x_max - tile_width) - x_min

        # Filter transcripts and labels
        tx_local_x = tx_geo_df[
            tx_geo_df["x_location"].between(
                x_start, x_start + tile_width, inclusive="left"
            )
        ]

        for y_start in np.arange(y_min, y_max + 1, tile_stride):
            # Handle edge cases
            y_start = min(y_start, y_max - tile_height) - y_min

            # Filter transcripts and labels
            tx_local = tx_local_x[
                tx_local_x["y_location"].between(
                    y_start, y_start + tile_height, inclusive="left"
                )
            ]

            # Save a numpy array of pixel x, pixel y, and gene ID
            X = tx_local["x_location"].values.astype(int) - x_min - x_start
            Y = tx_local["y_location"].values.astype(int) - y_min - y_start
            G = tx_local["gene_id"].values.astype(int)
            np.savez(f"{outdir}/tiles/transcripts/{image_id}", np.array([X, Y, G]).T)

            # Save a numpy matrix with every entry being the ID of the cell, background (0), or unknown (-1)
            labels_local = np.array(
                labels[x_start : x_start + tile_width, y_start : y_start + tile_height]
            )
            local_ids = np.unique(labels_local)
            local_ids = local_ids[local_ids > 0]
            for i, c in enumerate(local_ids):
                labels_local[labels_local == c] = i + 1
            np.savez(f"{outdir}/tiles/labels/{image_id}", labels_local)

            # Save a numpy matrix with every entry being the angle from this pixel to the centroid
            # of the nucleus (in [0,1]) or unknown (-1)
            angles_local = np.array(
                angles[x_start : x_start + tile_width, y_start : y_start + tile_height]
            )
            angles_local[labels_local == -1] = -1
            np.savez(f"{outdir}/tiles/angles/{image_id}", angles_local)

            # Save a numpy matrix with every entry being the class ID (cell type) of the cell (1...K),
            # background (0), or unknown (-1)
            classes_local = np.array(
                pixel_types[
                    x_start : x_start + tile_width, y_start : y_start + tile_height
                ]
            )
            np.savez(f"{outdir}/tiles/classes/{image_id}", classes_local)

            # Track the original coordinates that this tile belongs to
            tile_locations.append([x_start, y_start])

            # Track how many transcripts this tile has
            tx_local_counts.append(len(G))

            # Track how many of each class label this tile has
            temp = np.zeros(n_classes + 2, dtype=int)
            uniques = np.unique(classes_local, return_counts=True)
            logger.debug(f"uniques: {uniques}")
            temp[uniques[0] + 1] = uniques[1]
            class_local_counts.append(temp)

            # Update the image filename ID
            image_id += 1
            progress_bar.update(1)

    # Save the tile (x,y) locations
    logger.info(f"Saving {outdir}/tiles/locations.npy")
    np.save(f"{outdir}/tiles/locations.npy", np.array(tile_locations))
    logger.info(f"Saving {outdir}/tiles/class_counts.npy")
    np.save(f"{outdir}/tiles/class_counts.npy", class_local_counts)
    logger.info(f"Saving {outdir}/tiles/transcript_counts.npy")
    np.save(f"{outdir}/tiles/transcript_counts.npy", tx_local_counts)
    logger.info(f"Saving {outdir}/tiles/class_counts.npy")
    np.save(f"{outdir}/tiles/gene_ids.npy", np.array(list(enumerate(gene_ids))))
