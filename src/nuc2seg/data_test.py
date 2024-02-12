import shutil
import pytest
from nuc2seg.data import Nuc2SegDataset, TiledDataset, generate_tiles
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

    assert td[0] == {}
