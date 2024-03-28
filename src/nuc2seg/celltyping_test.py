from nuc2seg.celltyping import (
    fit_celltype_em_model,
    run_cell_type_estimation,
    select_best_celltyping_chain,
    create_dense_gene_counts_matrix,
    predict_celltypes_for_segments_and_transcripts,
)
import numpy as np
import geopandas
from shapely import Polygon, Point


def test_fit_celltype_em_model():
    rng = np.random.default_rng(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    celltyping_results = fit_celltype_em_model(
        gene_counts,
        gene_names=[f"gene_{i}" for i in range(n_genes)],
        min_components=2,
        max_components=10,
        max_em_steps=3,
        tol=1e-4,
        warm_start=False,
        rng=rng,
    )

    (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        relative_expression,
    ) = (
        celltyping_results.aic_scores,
        celltyping_results.bic_scores,
        celltyping_results.final_expression_profiles,
        celltyping_results.final_prior_probs,
        celltyping_results.final_cell_types,
        celltyping_results.relative_expression,
    )

    assert len(aic_scores) == 9
    assert len(bic_scores) == 9
    assert len(final_expression_profiles) == 9
    assert len(final_prior_probs) == 9
    assert len(final_cell_types) == 9
    for i, x in enumerate(final_cell_types):
        assert x.shape[0] == 100
        assert x.shape[1] == i + 2


def test_estimate_cell_types2():
    rng = np.random.default_rng(0)
    n_genes, n_cells = 12, 99

    data = np.zeros((n_cells, n_genes), dtype=int)
    data[:33, :4] = np.random.poisson(10, size=(33, 4))
    data[33:, :4] = np.random.poisson(1, size=(66, 4))
    data[33:66, 4:8] = np.random.poisson(10, size=(33, 4))
    data[:33, 4:8] = np.random.poisson(1, size=(33, 4))
    data[66:, :8] = np.random.poisson(1, size=(33, 8))

    data[66:, 8:] = np.random.poisson(10, size=(33, 4))
    data[:66, 8:] = np.random.poisson(1, size=(66, 4))

    celltyping_results = fit_celltype_em_model(
        data,
        gene_names=[f"gene_{i}" for i in range(n_genes)],
        min_components=2,
        max_components=25,
        max_em_steps=10,
        tol=1e-4,
        warm_start=False,
        rng=rng,
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

    assert len(aic_scores) == 24
    assert len(bic_scores) == 24
    assert aic_scores.argmin() == bic_scores.argmin() == 1


def test_run_cell_type_estimation(test_nuclei_df, test_transcripts_df):
    rng = np.random.default_rng(0)
    results = run_cell_type_estimation(
        nuclei_geo_df=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        foreground_nucleus_distance=1,
        max_components=3,
        min_components=2,
        rng=rng,
    )

    assert len(results.expression_profiles) == 2
    assert len(results.prior_probs) == 2
    assert len(results.relative_expression) == 2
    assert len(results.aic_scores) == 2
    assert len(results.bic_scores) == 2
    assert results.n_component_values.tolist() == [2, 3]


def test_combine_celltyping_chains(test_nuclei_df, test_transcripts_df):
    rng = np.random.default_rng(0)
    results = run_cell_type_estimation(
        nuclei_geo_df=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        foreground_nucleus_distance=1,
        max_components=3,
        min_components=2,
        rng=rng,
    )
    rng = np.random.default_rng(1)
    results2 = run_cell_type_estimation(
        nuclei_geo_df=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        foreground_nucleus_distance=1,
        max_components=3,
        min_components=2,
        rng=rng,
    )

    results.bic_scores = np.array([1.0, 2.0])
    results2.bic_scores = np.array([2.0, -1.0])

    final_result, _, _, best_k = select_best_celltyping_chain([results, results2])

    assert best_k == 1
    assert final_result is results2


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
        gene_id_col="gene_id",
    )

    np.testing.assert_array_equal(result, np.array([[1, 0], [1, 1]]))


def test_predict_celltypes_for_segments_and_transcripts():
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

    predict_celltypes_for_segments_and_transcripts(
        celltype_results=None,
        k=0,
        segment_geo_df=boundaries,
        transcript_geo_df=transcripts,
        chunk_size=1,
        gene_name_column="gene_id",
    )
