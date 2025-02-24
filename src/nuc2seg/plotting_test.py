import os.path
import shutil
import tempfile
import geopandas
import shapely

import numpy as np

from nuc2seg.celltyping import fit_celltype_em_model
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
from nuc2seg.plotting import (
    plot_model_predictions,
    plot_celltype_estimation_results,
    rank_genes_groups_plot,
    plot_greedy_cell_segmentation,
)
from nuc2seg.preprocessing import cart2pol


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
        classes=np.ones((64, 64)),
        transcripts=np.array([[0, 0, 0], [32, 32, 1], [35, 35, 2], [22, 22, 2]]),
        bbox=np.array([0, 0, 64, 64]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    predictions = ModelPredictions(
        angles=angles,
        classes=np.ones((64, 64, 3)),
        foreground=np.random.random((64, 64)),
    )

    prior_segmentation_gdf = geopandas.GeoDataFrame(
        {"segment_id": [1], "geometry": [shapely.box(26, 22, 38, 42)]}
    )

    segmentation_gdf = geopandas.GeoDataFrame(
        {"segment_id": [1], "geometry": [shapely.box(22, 10, 42, 50)]}
    )

    output_path = "test.pdf"

    tmpdir = tempfile.mkdtemp()

    try:
        plot_model_predictions(
            prior_segmentation_gdf=prior_segmentation_gdf,
            segmentation_gdf=segmentation_gdf,
            dataset=ds,
            model_predictions=predictions,
            output_path=os.path.join(tmpdir, output_path),
            bbox=shapely.box(1, 5, 63, 63),
        )
    finally:
        shutil.rmtree(tmpdir)


def test_plot_celltype_estimation_results():
    np.random.seed(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    tmpdir = tempfile.mkdtemp()

    try:
        celltyping_results = fit_celltype_em_model(
            gene_counts,
            gene_names=np.array([f"gene_{i}" for i in range(n_genes)]),
            min_components=2,
            max_components=4,
            max_em_steps=3,
            tol=1e-4,
            warm_start=False,
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

        plot_celltype_estimation_results(
            np.stack([aic_scores * 1.01, aic_scores, aic_scores * 0.99]),
            np.stack([bic_scores * 1.01, bic_scores, bic_scores * 0.99]),
            final_expression_profiles,
            final_prior_probs,
            relative_expression,
            celltyping_results.n_component_values,
            tmpdir,
        )
    finally:
        shutil.rmtree(tmpdir)


def test_rank_genes_groups_plot():
    np.random.seed(0)
    n_genes, n_cells = 10, 100

    gene_counts = np.random.poisson(10, size=(n_cells, n_genes))

    tmpdir = tempfile.mkdtemp()

    try:
        celltyping_results = fit_celltype_em_model(
            gene_counts,
            gene_names=np.array([f"gene_{i}" for i in range(n_genes)]),
            min_components=2,
            max_components=4,
            max_em_steps=3,
            tol=1e-4,
            warm_start=False,
        )
        rank_genes_groups_plot(
            celltyping_results,
            k=4,
            output_path=os.path.join(tmpdir, "rank_genes_groups.pdf"),
        )

    finally:
        shutil.rmtree(tmpdir)


def test_plot_greedy_cell_segmentation():
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
        labels=labels.astype(int),
        angles=angles,
        classes=np.ones((64, 64)).astype(float),
        transcripts=np.array([[30, 11, 1], [30, 13, 0], [30, 20, 0], [30, 21, 0]]),
        bbox=np.array([0, 0, 64, 64]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    predictions = ModelPredictions(
        angles=angles,
        classes=np.ones((64, 64, 3)).astype(float),
        foreground=np.ones_like(labels).astype(float),
    )

    output_dir = tempfile.mkdtemp()

    try:
        plot_greedy_cell_segmentation(
            dataset=ds,
            predictions=predictions,
            prior_probs=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
            expression_profiles=np.array(
                [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.98, 0.01, 0.01]]
            ),
            output_path=os.path.join(output_dir, "test.gif"),
            segment_id=1,
        )
    finally:
        shutil.rmtree(output_dir)
