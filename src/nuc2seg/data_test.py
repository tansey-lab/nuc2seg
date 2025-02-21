import os.path
import shutil
import tempfile

import numpy as np
from blended_tiling import TilingModule

from nuc2seg.conftest import test_dataset
from nuc2seg.data import (
    Nuc2SegDataset,
    TiledDataset,
    CelltypingResults,
    RasterizedDataset,
    ModelPredictions,
)
from nuc2seg.utils import generate_tiles, get_indexed_tiles


def test_Nuc2SegDataset():
    ds = Nuc2SegDataset(
        labels=np.ones((10, 20)),
        angles=np.ones((10, 20)),
        classes=np.ones((10, 20, 3)),
        transcripts=np.array([[0, 0, 0], [5, 5, 1], [10, 10, 2]]),
        bbox=np.array([100, 100, 110, 120]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "test.h5")

    assert ds.n_classes == 3
    assert ds.n_genes == 3
    assert ds.x_extent_pixels == 10
    assert ds.y_extent_pixels == 20

    try:
        ds.save_h5(output_path)
        ds2 = Nuc2SegDataset.load_h5(output_path)

        np.testing.assert_array_equal(ds.labels, ds2.labels)
        np.testing.assert_array_equal(ds.angles, ds2.angles)
        np.testing.assert_array_equal(ds.classes, ds2.classes)
        np.testing.assert_array_equal(ds.transcripts, ds2.transcripts)
        np.testing.assert_array_equal(ds.bbox, ds2.bbox)
        assert ds.n_classes == ds2.n_classes
        assert ds.n_genes == ds2.n_genes

        ds_clipped = ds.clip(np.array([0, 0, 10, 10]))
        assert ds_clipped.x_extent_pixels == 10
        assert ds_clipped.y_extent_pixels == 10
        assert ds_clipped.labels.shape == (10, 10)
        np.testing.assert_array_equal(ds_clipped.bbox, np.array([100, 100, 110, 110]))
    finally:
        shutil.rmtree(tmpdir)


def test_Nuc2SegDataset_load_tile():
    ds = Nuc2SegDataset(
        labels=np.ones((10, 10)),
        angles=np.ones((10, 10)),
        classes=np.ones((10, 10, 3)),
        transcripts=np.array([[0, 0, 0], [9, 9, 2]]),
        bbox=np.array([100, 100, 110, 110]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "test.h5")
    ds.save_h5(output_path)

    tile_idx_lookup = get_indexed_tiles(
        extent=(10, 10),
        tile_size=(5, 5),
        overlap=0.2,
    )

    ds2 = Nuc2SegDataset.load_h5(
        output_path, tile_width=5, tile_height=5, tile_overlap=0.2, tile_index=0
    )

    ds3 = Nuc2SegDataset.load_h5(
        output_path, tile_width=5, tile_height=5, tile_overlap=0.2, tile_index=8
    )

    try:
        assert ds2.x_extent_pixels == 5
        assert ds2.y_extent_pixels == 5
        assert len(ds2.transcripts) == 1
        np.testing.assert_almost_equal(ds2.bbox, np.array([100, 100, 105, 105]))

        assert ds3.x_extent_pixels == 5
        assert ds3.y_extent_pixels == 5
        assert len(ds3.transcripts) == 1
        assert tuple(ds3.transcripts[0]) == (4, 4, 2)
        np.testing.assert_almost_equal(ds3.bbox, np.array([105, 105, 110, 110]))
    finally:
        shutil.rmtree(tmpdir)


def test_generate_tiles():
    tiler = TilingModule(
        base_size=(10, 20), tile_size=(5, 10), tile_overlap=(0.25, 0.25)
    )

    tile_bboxes = list(
        generate_tiles(
            tiler=tiler,
            x_extent=10,
            y_extent=20,
            tile_size=(5, 10),
            overlap_fraction=0.25,
        )
    )
    assert len(tile_bboxes) == 9


def test_tiled_dataset(test_dataset):
    td = TiledDataset(
        dataset=test_dataset, tile_height=10, tile_width=10, tile_overlap=0.25
    )

    assert len(td) == 9

    first_tile = td[0]

    assert first_tile["angles"].shape == (10, 10)
    assert first_tile["labels"].shape == (10, 10)
    assert first_tile["classes"].shape == (10, 10)
    assert first_tile["nucleus_mask"].shape == (10, 10)

    assert td.per_tile_class_histograms.shape == (len(td), test_dataset.n_classes + 2)

    second_tile = td[1]

    assert second_tile["angles"].shape == (10, 10)
    assert second_tile["labels"].shape == (10, 10)
    assert second_tile["classes"].shape == (10, 10)
    assert second_tile["nucleus_mask"].shape == (10, 10)


def test_celltype_results():
    results = CelltypingResults(
        aic_scores=np.array([1, 2, 3]),
        bic_scores=np.array([1, 2, 3]),
        expression_profiles=[
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ],
        prior_probs=[
            np.array([[1, 2]]),
            np.array([[1, 2, 3]]),
            np.array([[1, 2, 3, 4]]),
        ],
        relative_expression=[
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ],
        min_n_components=2,
        max_n_components=4,
        gene_names=np.array(["gene1", "gene2", "gene3"]),
    )

    tmpdir = tempfile.mkdtemp()

    try:
        results.save_h5(os.path.join(tmpdir, "celltype_results.h5"))
        results2 = CelltypingResults.load_h5(
            os.path.join(tmpdir, "celltype_results.h5")
        )

        np.testing.assert_array_equal(results.aic_scores, results2.aic_scores)
        np.testing.assert_array_equal(results.bic_scores, results2.bic_scores)
        assert len(results.expression_profiles) == len(results2.expression_profiles)
        assert len(results.prior_probs) == len(results2.prior_probs)
        assert len(results.relative_expression) == len(results2.relative_expression)
    finally:
        shutil.rmtree(tmpdir)


def test_rasterized_dataset():
    ds = RasterizedDataset(
        labels=np.ones((10, 20)),
        angles=np.ones((10, 20)),
        transcripts=np.array([[0, 0, 0], [5, 5, 1], [10, 10, 2]]),
        bbox=np.array([100, 100, 110, 120]),
        n_genes=3,
        resolution=1,
    )

    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "test.h5")

    assert ds.n_genes == 3
    assert ds.x_extent_pixels == 10
    assert ds.y_extent_pixels == 20

    try:
        ds.save_h5(output_path)
        ds2 = RasterizedDataset.load_h5(output_path)

        np.testing.assert_array_equal(ds.labels, ds2.labels)
        np.testing.assert_array_equal(ds.angles, ds2.angles)
        np.testing.assert_array_equal(ds.transcripts, ds2.transcripts)
        np.testing.assert_array_equal(ds.bbox, ds2.bbox)
        assert ds.n_genes == ds2.n_genes
    finally:
        shutil.rmtree(tmpdir)


def test_get_background_frequencies():
    labels = np.array([[0, 0, 1], [0, -1, 1], [1, 1, 1]])
    angles = np.random.random((3, 3))
    transcripts = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 2], [2, 2, 2]])

    ds = Nuc2SegDataset(
        labels=labels,
        angles=angles,
        transcripts=transcripts,
        bbox=np.array([100, 100, 102, 102]),
        n_genes=3,
        resolution=1,
        n_classes=2,
        classes=np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]]),
    )

    background_frequencies = ds.get_background_frequencies().detach().cpu().numpy()

    assert len(background_frequencies) == 3
    np.testing.assert_almost_equal(
        background_frequencies, np.array([0, 0.66666, 0.33333]), decimal=3
    )


def test_get_class_frequencies():
    labels = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]])
    angles = np.random.random((3, 3))
    transcripts = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 2], [0, 1, 2], [2, 2, 2]])
    classes = np.array([[0, 1, 1], [1, 1, 1], [2, 2, 2]])

    ds = Nuc2SegDataset(
        labels=labels,
        angles=angles,
        transcripts=transcripts,
        bbox=np.array([100, 100, 102, 102]),
        n_genes=3,
        resolution=1,
        n_classes=2,
        classes=classes,
    )

    celltype_frequencies = ds.get_celltype_frequencies().detach().cpu().numpy()

    np.testing.assert_almost_equal(
        celltype_frequencies,
        np.array(
            [
                [0, 0, 0.2],
                [0, 0, 0.33333],
            ]
        ),
        decimal=3,
    )


def test_ModelPredictions_load_tile():
    data = ModelPredictions(
        angles=np.ones((10, 10)),
        classes=np.ones((10, 10, 3)),
        foreground=np.ones((10, 10)),
    )

    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "test.h5")
    data.save_h5(output_path)

    data2 = ModelPredictions.load_h5(
        output_path, tile_width=5, tile_height=5, tile_overlap=0.2, tile_index=0
    )

    try:
        assert data2.angles.shape == (5, 5)
    finally:
        shutil.rmtree(tmpdir)
