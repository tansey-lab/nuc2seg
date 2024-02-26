from nuc2seg.celltyping import estimate_cell_types
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
