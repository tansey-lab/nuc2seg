from nuc2seg.plotting import plot_model_predictions, plot_celltype_estimation_results
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
from nuc2seg.celltyping import estimate_cell_types
from nuc2seg.preprocessing import cart2pol
import numpy as np
import tempfile
import shutil
import os.path


def test_plot_model_predictions():
    labels = np.zeros((64, 64))

    labels[22:42, 10:50] = -1
    labels[26:38, 22:42] = 1

    angles = np.zeros((64, 64))

    for x in range(64):
        for y in range(64):
            x_component = 32 - x
            y_component = 32 - y
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    ds = Nuc2SegDataset(
        labels=labels,
        angles=angles,
        classes=np.ones((10, 20)),
        transcripts=np.array([[0, 0, 0], [32, 32, 1], [35, 35, 2], [22, 22, 2]]),
        bbox=np.array([0, 0, 64, 64]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    predictions = ModelPredictions(
        angles=angles,
        classes=np.ones((10, 20)),
        foreground=np.random.random((64, 64)),
    )

    output_path = "test.png"

    tmpdir = tempfile.mkdtemp()

    try:
        plot_model_predictions(
            ds, predictions, os.path.join(tmpdir, output_path), bbox=[1, 1, 63, 63]
        )
        plot_model_predictions(
            ds,
            predictions,
            os.path.join(tmpdir, "quiver.png"),
            use_quiver=True,
            bbox=[1, 1, 63, 63],
        )
    finally:
        shutil.rmtree(tmpdir)


def test_plot_celltype_estimation_results():
    np.random.seed(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    tmpdir = tempfile.mkdtemp()

    try:
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

        plot_celltype_estimation_results(
            aic_scores,
            bic_scores,
            final_expression_profiles,
            final_prior_probs,
            final_cell_types,
            relative_expression,
            tmpdir,
        )
    finally:
        shutil.rmtree(tmpdir)
