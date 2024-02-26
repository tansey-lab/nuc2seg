from nuc2seg.celltyping import (
    estimate_cell_types,
    run_cell_type_estimation,
    combine_celltyping_chains,
)
import numpy as np


def test_estimate_cell_types():
    np.random.seed(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    celltyping_results = estimate_cell_types(
        gene_counts,
        min_components=2,
        max_components=10,
        max_em_steps=3,
        tol=1e-4,
        warm_start=False,
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
    np.random.seed(0)
    n_genes, n_cells = 12, 99

    data = np.zeros((n_cells, n_genes), dtype=int)
    data[:33, :4] = np.random.poisson(10, size=(33, 4))
    data[33:, :4] = np.random.poisson(1, size=(66, 4))
    data[33:66, 4:8] = np.random.poisson(10, size=(33, 4))
    data[:33, 4:8] = np.random.poisson(1, size=(33, 4))
    data[66:, :8] = np.random.poisson(1, size=(33, 8))

    data[66:, 8:] = np.random.poisson(10, size=(33, 4))
    data[:66, 8:] = np.random.poisson(1, size=(66, 4))

    celltyping_results = estimate_cell_types(
        data,
        min_components=2,
        max_components=25,
        max_em_steps=10,
        tol=1e-4,
        warm_start=False,
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

    assert len(results.final_cell_types) == 2
    assert len(results.final_expression_profiles) == 2
    assert len(results.final_prior_probs) == 2
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

    final_result, _, _ = combine_celltyping_chains([results, results2])

    np.testing.assert_array_equal(
        final_result.aic_scores,
        np.stack([results.aic_scores, results2.aic_scores]).mean(axis=0),
    )

    np.testing.assert_array_equal(
        final_result.bic_scores,
        np.stack([results.bic_scores, results2.bic_scores]).mean(axis=0),
    )
