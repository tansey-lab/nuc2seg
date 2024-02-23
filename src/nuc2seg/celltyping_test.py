from nuc2seg.celltyping import estimate_cell_types
import numpy as np


def test_estimate_cell_types():
    np.random.seed(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    (
        aic_scores,
        bic_scores,
        final_expression_profiles,
        final_prior_probs,
        final_cell_types,
        relative_expression,
    ) = estimate_cell_types(
        gene_counts,
        min_components=2,
        max_components=10,
        max_em_steps=3,
        tol=1e-4,
        warm_start=False,
    )

    assert len(aic_scores) == 9
    assert len(bic_scores) == 9
    assert len(final_expression_profiles) == 9
    assert len(final_prior_probs) == 9
    assert len(final_cell_types) == 9
    for i, x in enumerate(final_cell_types):
        assert x.shape[0] == 100
        assert x.shape[1] == i + 2
