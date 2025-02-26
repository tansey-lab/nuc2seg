import glob
import os.path
import shutil
import tempfile

import numpy as np
import pandas
import shapely

from nuc2seg.preprocessing import (
    create_rasterized_dataset,
    tile_transcripts_to_disk,
    create_pixel_geodf,
)


def test_create_rasterized_dataset(test_nuclei_df, test_transcripts_df):
    np.random.seed(0)
    ds = create_rasterized_dataset(
        prior_segmentation_gdf=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        sample_area=shapely.Polygon([(1, 1), (30, 1), (30, 20), (1, 20), (1, 1)]),
        resolution=1,
        foreground_distance=1,
        background_distance=4,
        background_transcript_distance=2,
        background_pixel_transcripts=2,
    )

    assert ds.labels.shape == (29, 19)
    assert ds.transcripts.shape == (8, 3)
    assert ds.x_extent_pixels == 29
    assert ds.y_extent_pixels == 19
    assert ds.n_genes == 2

    # Assert coordinated are transformed relative to the bbox
    assert ds.transcripts[:, 0].min() == 9.0
    assert ds.transcripts[:, 1].min() == 8.0
    assert sorted(np.unique(ds.labels).tolist()) == [-1, 0, 1, 2]


def test_tile_transcripts_to_csv(test_transcripts_df):
    output_dir = tempfile.mkdtemp()

    try:
        tile_transcripts_to_disk(
            transcripts=test_transcripts_df,
            tile_size=(10, 10),
            overlap=0.5,
            output_dir=output_dir,
            output_format="csv",
            bounds=(0, 0, 20, 20),
        )
        output_fns = list(glob.glob(os.path.join(output_dir, "*.csv")))

        assert len(output_fns) == 9
        for fn in output_fns:
            df = pandas.read_csv(fn)
    finally:
        shutil.rmtree(output_dir)


def test_create_pixel_geodf():
    result = create_pixel_geodf(
        width=12,
        height=12,
        resolution=5,
    )
    assert tuple(result.total_bounds) == (0, 0, 15, 15)
    assert len(result) == 9
