import shutil

from nuc2seg.data import Nuc2SegDataset
import numpy as np
import tempfile
import os.path


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
