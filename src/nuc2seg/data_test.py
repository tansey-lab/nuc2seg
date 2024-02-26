import shutil
import pytest
from nuc2seg.data import (
    Nuc2SegDataset,
    TiledDataset,
    generate_tiles,
    CelltypingResults,
    RasterizedDataset,
)
import numpy as np
import tempfile
import os.path
from blended_tiling import TilingModule


@pytest.fixture(scope="package")
def test_dataset():
    return Nuc2SegDataset(
        labels=np.ones((10, 20)),
        angles=np.ones((10, 20)),
        classes=np.ones((10, 20)),
        transcripts=np.array([[0, 0, 0], [5, 5, 1], [10, 10, 2]]),
        bbox=np.array([100, 100, 110, 120]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )


def test_Nuc2SegDataset():
    ds = Nuc2SegDataset(
        labels=np.ones((10, 20)),
        angles=np.ones((10, 20)),
        classes=np.ones((10, 20)),
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
        dataset=test_dataset, tile_height=10, tile_width=5, tile_overlap=0.25
    )

    assert len(td) == 9

    first_tile = td[0]

    assert first_tile["angles"].shape == (5, 10)
    assert first_tile["labels"].shape == (5, 10)
    assert first_tile["classes"].shape == (5, 10)
    assert first_tile["location"].size == 2
    assert first_tile["nucleus_mask"].shape == (5, 10)

    assert td.per_tile_class_histograms.shape == (len(td), test_dataset.n_classes + 2)

    second_tile = td[1]

    assert second_tile["angles"].shape == (5, 10)
    assert second_tile["labels"].shape == (5, 10)
    assert second_tile["classes"].shape == (5, 10)
    assert second_tile["location"].size == 2
    assert second_tile["nucleus_mask"].shape == (5, 10)


def test_celltype_results():
    results = CelltypingResults(
        aic_scores=np.array([1, 2, 3]),
        bic_scores=np.array([1, 2, 3]),
        final_expression_profiles=[
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ],
        final_prior_probs=[
            np.array([[1, 2]]),
            np.array([[1, 2, 3]]),
            np.array([[1, 2, 3, 4]]),
        ],
        final_cell_types=[
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ],
        relative_expression=[
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ],
        min_n_components=2,
        max_n_components=4,
    )

    tmpdir = tempfile.mkdtemp()

    try:
        results.save_h5(os.path.join(tmpdir, "celltype_results.h5"))
        results2 = CelltypingResults.load_h5(
            os.path.join(tmpdir, "celltype_results.h5")
        )

        np.testing.assert_array_equal(results.aic_scores, results2.aic_scores)
        np.testing.assert_array_equal(results.bic_scores, results2.bic_scores)
        assert len(results.final_expression_profiles) == len(
            results2.final_expression_profiles
        )
        assert len(results.final_prior_probs) == len(results2.final_prior_probs)
        assert len(results.final_cell_types) == len(results2.final_cell_types)
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
