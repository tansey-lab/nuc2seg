#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial Transcriptomics Analysis using scVI and KDTree

This script performs spatial transcriptomics analysis on nucleus data using scVI for
dimensionality reduction and KDTree for spatial queries of transcripts.
"""

import numpy as np
import anndata
import scvi
import tqdm
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from nuc2seg.xenium import load_and_filter_transcripts_as_points, load_vertex_file
from nuc2seg.preprocessing import create_pixel_geodf
from nuc2seg.postprocess import convert_transcripts_to_anndata
import pandas


def get_transcript_density(transcript_xy, pixel_xy, grid_shape, bandwidth=0.5):
    df = pandas.DataFrame({"x": transcript_xy[:, 0], "y": transcript_xy[:, 1]})
    df["x"] = df["x"].clip(0, grid_shape[0] - 1).astype(int)
    df["y"] = df["y"].clip(0, grid_shape[1] - 1).astype(int)
    df = df.groupby(["x", "y"]).size().reset_index(name="count")

    # transform back to grid
    x = df["x"].values
    y = df["y"].values
    output = np.zeros((grid_shape[0], grid_shape[1]))
    output[x, y] = df["count"].values

    return output


def categorical_to_sparse(distances, categories):
    """
    Convert categorical data with distances to a sparse matrix with weighted counts.

    Parameters:
    ----------
    distances : numpy.ndarray
        Distance matrix for each category
    categories : numpy.ndarray
        Array of categorical indices

    Returns:
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of weighted category counts
    """
    # Set minimum distance to 1.0 to avoid division by zero
    distances[distances < 1.0] = 1.0

    # Get dimensions and unique categories
    n_rows = categories.shape[0]
    n_categories = np.max(categories) + 1

    # Initialize lists to construct COO format
    row_indices = []
    col_indices = []
    data_values = []

    # Count occurrences of each category in each row
    for row_idx in tqdm.trange(n_rows):
        row_data = categories[row_idx]
        weight = 1.0 / distances[row_idx]
        onehot = np.zeros((len(row_data), n_categories), dtype=int)
        onehot[np.arange(len(row_data)), row_data] = 1

        onehot = onehot * weight[:, None]
        counts = np.sum(onehot, axis=0)

        for col_idx, val in enumerate(counts):
            if val > 0:
                # Add to our COO format lists
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                data_values.append(val)

    # Create sparse matrix in CSR format
    sparse_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)), shape=(n_rows, n_categories)
    )

    return sparse_matrix


def normalize_to_255(data):
    """
    Normalize data to 0-255 range.

    Parameters:
    ----------
    data : numpy.ndarray
        Array to normalize

    Returns:
    -------
    numpy.ndarray
        Normalized array with values between 0-255
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # Handle the special case where all values are the same
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8)

    # Normalize to 0-1 range
    normalized = (data - min_val) / (max_val - min_val)

    # Scale to 0-255 range and convert to uint8
    normalized_255 = (normalized * 255).astype(np.uint8)

    return normalized_255


def main():
    nn_k = 30
    # Load and prepare transcript data
    tx = load_and_filter_transcripts_as_points(
        "~/Code/nuc2seg/nextflow/tests/data/prostate/transcripts.parquet"
    )
    tx["geometry"] = tx.translate(-2000, -2000)  # Translate coordinates
    tx = tx.reset_index(drop=True)

    nuclei = load_vertex_file(
        "~/Code/nuc2seg/nextflow/tests/data/prostate/nucleus_boundaries.parquet"
    )
    nuclei["geometry"] = nuclei.translate(-2000, -2000)  # Translate coordinates
    nuclei = nuclei.reset_index(drop=True)

    cells = load_vertex_file(
        "~/Code/nuc2seg/nextflow/tests/data/prostate/cell_boundaries.parquet"
    )
    cells["geometry"] = cells.translate(-2000, -2000)  # Translate coordinates
    cells = cells.reset_index(drop=True)

    # Create pixel grid for spatial analysis
    pixel_geodf = create_pixel_geodf(400, 400)

    # Set up KD-tree for spatial queries
    transcript_xy = np.array(
        [
            tx.centroid.x.values,
            tx.centroid.y.values,
        ]
    ).T
    kdtree = KDTree(transcript_xy)

    # Query nearest transcripts for each pixel
    pixels_xy = np.array([pixel_geodf.centroid.x, pixel_geodf.centroid.y]).T
    kdtree_lookup = kdtree.query(pixels_xy, k=nn_k)

    print("Computing transcript density")
    density = get_transcript_density(transcript_xy, pixels_xy, (400, 400))

    adata = convert_transcripts_to_anndata(
        transcript_gdf=tx,
        segmentation_gdf=nuclei,
        gene_name_column="feature_name",
        min_molecules_per_cell=10,
    )

    # Set up and train scVI model
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=None,  # use raw counts
        batch_key=None,  # no batch correction in this example
    )
    n_latent = 5
    model = scvi.model.SCVI(
        adata,
        n_hidden=128,
        n_latent=n_latent,  # latent space dimension
        n_layers=2,
        dropout_rate=0.1,
    )

    # Train the model
    model.train(max_epochs=400, early_stopping=True, early_stopping_patience=20)

    # Get latent representation
    adata.obsm["X_scVI"] = model.get_latent_representation()

    # Extract gene IDs for each nearest transcript
    gene_ids = tx.gene_id.values[np.ravel(kdtree_lookup[1])].reshape(
        kdtree_lookup[1].shape
    )

    # Convert to sparse matrix representation
    sparse_pixel_data = categorical_to_sparse(kdtree_lookup[0], gene_ids)

    # Create anndata object for pixels
    pixel_adata = anndata.AnnData(X=sparse_pixel_data, var=adata.var)
    pixel_adata.obsm["spatial"] = pixels_xy

    # Project pixels into latent space using trained model
    vae_q = model.load_query_data(pixel_adata, model)
    vae_q.is_trained = True
    pixel_adata.obsm["X_scVI"] = vae_q.get_latent_representation()

    # Create RGB image from latent space
    output_rgb = np.zeros((400, 400, n_latent))
    for i in range(n_latent):
        output_rgb[:, :, i] = pixel_adata.obsm["X_scVI"][:, i].reshape(400, 400)
    # Display the result
    fig, ax = plt.subplots(figsize=(10, 10 * n_latent), nrows=n_latent + 1, ncols=1)

    for i in range(n_latent):
        ax[i].imshow(output_rgb[:, :, i].T, cmap="coolwarm")
        ax[i].set_title(f"Latent Dimension {i}")

        nuclei.plot(ax=ax[i], facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=0.7)
        cells.plot(ax=ax[i], facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=0.3)

    im = ax[n_latent].imshow(np.log1p(density.T), cmap="coolwarm")
    ax[n_latent].set_title("Log Transcript Density")
    nuclei.plot(
        ax=ax[n_latent], facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=0.7
    )
    cells.plot(
        ax=ax[n_latent], facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=0.3
    )
    # add colorbar
    cbar = fig.colorbar(im, ax=ax[n_latent])
    cbar.set_label("Transcript Density")
    plt.tight_layout()

    plt.savefig("/tmp/spatial_transcriptomics_visualization.pdf", dpi=300)
