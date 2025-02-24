import geopandas
import numpy as np
import pytest
import torch
from shapely import Polygon, Point, box
from nuc2seg.constants import NOISE_CELLTYPE

from nuc2seg.celltyping import (
    fit_celltype_em_model,
    predict_celltypes_for_anndata_with_noise_type,
    select_best_celltyping_chain,
    create_dense_gene_counts_matrix,
    predict_celltypes_for_segments_and_transcripts,
    predict_celltype_probabilities_for_all_segments,
)


def test_fit_celltype_em_model():
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    celltyping_results = fit_celltype_em_model(
        torch.tensor(gene_counts).int(),
        gene_names=[f"gene_{i}" for i in range(n_genes)],
        min_components=2,
        max_components=10,
        max_em_steps=3,
        tol=1e-4,
        warm_start=False,
        seed=42,
    )

    (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        relative_expression,
    ) = (
        celltyping_results.aic_scores,
        celltyping_results.bic_scores,
        celltyping_results.expression_profiles,
        celltyping_results.prior_probs,
        celltyping_results.relative_expression,
    )

    assert len(aic_scores) == 9
    assert len(bic_scores) == 9
    assert len(final_expression_profiles) == 9
    assert len(final_prior_probs) == 9


def test_estimate_cell_types2():
    rng = np.random.default_rng(0)
    n_genes, n_cells = 12, 99

    data = np.zeros((n_cells, n_genes), dtype=int)
    data[:33, :4] = rng.poisson(10, size=(33, 4))
    data[33:, :4] = rng.poisson(1, size=(66, 4))
    data[33:66, 4:8] = rng.poisson(10, size=(33, 4))
    data[:33, 4:8] = rng.poisson(1, size=(33, 4))
    data[66:, :8] = rng.poisson(1, size=(33, 8))

    data[66:, 8:] = rng.poisson(10, size=(33, 4))
    data[:66, 8:] = rng.poisson(1, size=(66, 4))

    celltyping_results = fit_celltype_em_model(
        torch.tensor(data).int(),
        gene_names=[f"gene_{i}" for i in range(n_genes)],
        min_components=2,
        max_components=5,
        max_em_steps=10,
        tol=1e-4,
        warm_start=False,
        seed=42,
    )

    (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        relative_expression,
    ) = (
        celltyping_results.aic_scores,
        celltyping_results.bic_scores,
        celltyping_results.expression_profiles,
        celltyping_results.prior_probs,
        celltyping_results.relative_expression,
    )

    assert len(aic_scores) == 4
    assert len(bic_scores) == 4
    assert aic_scores.argmin() == bic_scores.argmin() == 1


def test_create_dense_gene_counts_matrix():
    boundaries = geopandas.GeoDataFrame(
        [
            [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            [Polygon([(1, 0), (1, 1), (3, 1), (3, 0)])],
        ],
        columns=["geometry"],
    )

    transcripts = geopandas.GeoDataFrame(
        [
            ["a", Point(0.5, 0.5)],
            ["a", Point(2, 0.5)],
            ["b", Point(2, 0.5)],
        ],
        columns=["feature_name", "geometry"],
    )

    transcripts["gene_id"] = [0, 0, 1]

    result = create_dense_gene_counts_matrix(
        segmentation_geo_df=boundaries,
        transcript_geo_df=transcripts,
        n_genes=2,
        gene_id_col="gene_id",
    )

    np.testing.assert_array_equal(result, np.array([[1, 0], [1, 1]]))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_predict_celltypes_for_segments_and_transcripts(chunk_size):
    boundaries = geopandas.GeoDataFrame(
        [
            [box(0, 0, 1, 1)],
            [box(1, 1, 2, 2)],
            [box(2, 2, 3, 3)],
            [box(3, 3, 4, 4)],
        ],
        columns=["geometry"],
    )

    transcripts = geopandas.GeoDataFrame(
        [
            ["a", Point(0.5, 0.5)],
            ["b", Point(1.5, 1.5)],
            ["a", Point(2.5, 2.5)],
            ["b", Point(3.5, 3.5)],
        ],
        columns=["feature_name", "geometry"],
    )

    transcripts["gene_id"] = [0, 1, 0, 1]

    result = predict_celltypes_for_segments_and_transcripts(
        expression_profiles=np.array([[1, 1e-3], [1e-3, 1]]),
        prior_probs=np.array([0.6, 0.4]),
        segment_geo_df=boundaries,
        transcript_geo_df=transcripts,
        chunk_size=chunk_size,
        gene_name_column="gene_id",
    )

    assert result.argmax(axis=1).tolist() == [0, 1, 0, 1]


def test_predict_celltype_probabilities_for_all_segments():
    labels = np.array([[0, 1], [2, -1]])

    transcripts = np.array([[0, 1, 0], [1, 0, 1]])

    expression_profiles = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior_probs = np.array([0.6, 0.4])
    result = predict_celltype_probabilities_for_all_segments(
        labels, transcripts, expression_profiles, prior_probs
    )


def test_predict_celltypes_for_anndata_with_noise_type(test_adata):
    results = predict_celltypes_for_anndata_with_noise_type(
        ad=test_adata,
        prior_probs=np.array([0.6, 0.4]),
        expression_profiles=np.array([[1, 1e-3], [1e-3, 1]]),
    )

    assert (results.argmax(axis=1) + 1).tolist() == [NOISE_CELLTYPE, NOISE_CELLTYPE]

    test_adata = test_adata.copy()
    test_adata.X = test_adata.X * 10

    results = predict_celltypes_for_anndata_with_noise_type(
        ad=test_adata,
        prior_probs=np.array([0.6, 0.4]),
        expression_profiles=np.array([[1, 1e-3], [1e-3, 1]]),
    )

    assert (results.argmax(axis=1) + 1).tolist() == [2, 2]
