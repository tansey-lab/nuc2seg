import glob
import os.path
import shutil
import tempfile

import numpy as np
import pandas
import shapely
from shapely.affinity import translate

from nuc2seg.preprocessing import (
    create_rasterized_dataset,
    tile_transcripts_to_disk,
    create_pixel_geodf,
)
from nuc2seg.utils import transform_shapefile_to_rasterized_space


def test_create_rasterized_dataset(test_nuclei_df, test_transcripts_df):
    test_nuclei_df = test_nuclei_df.copy()
    test_transcripts_df = test_transcripts_df.copy()
    np.random.seed(0)
    sample_area = shapely.Polygon([(1, 1), (30, 1), (30, 20), (1, 20), (1, 1)])
    sample_area = translate(sample_area, xoff=10, yoff=10)
    test_nuclei_df["geometry"] = test_nuclei_df.geometry.translate(xoff=10, yoff=10)
    test_transcripts_df["geometry"] = test_transcripts_df.geometry.translate(
        xoff=10, yoff=10
    )
    background_pixel_transcripts = 2
    ds = create_rasterized_dataset(
        prior_segmentation_gdf=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        sample_area=sample_area,
        resolution=1,
        foreground_distance=1,
        background_distance=4,
        background_transcript_distance=2,
        background_pixel_transcripts=background_pixel_transcripts,
    )

    assert ds.labels.shape == (29, 19)
    assert ds.transcripts.shape == (8, 3)
    assert ds.x_extent_pixels == 29
    assert ds.y_extent_pixels == 19
    assert ds.n_genes == 2

    test_transcripts_df_rasterized = transform_shapefile_to_rasterized_space(
        test_transcripts_df, resolution=ds.resolution, sample_area=ds.bbox
    )

    for i in np.arange(ds.shape[0]):
        for j in np.arange(ds.shape[1]):
            x_filter = np.floor(test_transcripts_df_rasterized.centroid.x) == i
            y_filter = np.floor(test_transcripts_df_rasterized.centroid.y) == j
            if (x_filter & y_filter).sum() >= background_pixel_transcripts:
                assert ds.labels[i, j] != 0

    # Assert coordinated are transformed relative to the bbox
    assert ds.transcripts[:, 0].min() == 9.0
    assert ds.transcripts[:, 1].min() == 8.0
    assert sorted(np.unique(ds.labels).tolist()) == [-1, 0, 1]


def test_create_rasterized_dataset_with_resolution(test_nuclei_df, test_transcripts_df):
    test_nuclei_df = test_nuclei_df.copy()
    test_transcripts_df = test_transcripts_df.copy()
    np.random.seed(0)
    sample_area = shapely.Polygon([(1, 1), (30, 1), (30, 20), (1, 20), (1, 1)])
    sample_area = translate(sample_area, xoff=10, yoff=10)
    test_nuclei_df["geometry"] = test_nuclei_df.geometry.translate(xoff=10, yoff=10)
    test_transcripts_df["geometry"] = test_transcripts_df.geometry.translate(
        xoff=10, yoff=10
    )
    background_pixel_transcripts = 2
    ds = create_rasterized_dataset(
        prior_segmentation_gdf=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        sample_area=sample_area,
        resolution=0.5,
        foreground_distance=1,
        background_distance=4,
        background_transcript_distance=2,
        background_pixel_transcripts=background_pixel_transcripts,
    )

    assert ds.labels.shape == (58, 38)
    assert ds.transcripts.shape == (8, 3)
    assert ds.x_extent_pixels == 58
    assert ds.y_extent_pixels == 38
    assert ds.n_genes == 2

    test_transcripts_df_rasterized = transform_shapefile_to_rasterized_space(
        test_transcripts_df, resolution=ds.resolution, sample_area=ds.bbox
    )

    for i in np.arange(ds.shape[0]):
        for j in np.arange(ds.shape[1]):
            x_filter = np.floor(test_transcripts_df_rasterized.centroid.x) == i
            y_filter = np.floor(test_transcripts_df_rasterized.centroid.y) == j
            if (x_filter & y_filter).sum() >= background_pixel_transcripts:
                assert ds.labels[i, j] != 0

    # Assert coordinated are transformed relative to the bbox
    assert ds.transcripts[:, 0].min() == 18.0
    assert ds.transcripts[:, 1].min() == 17.0
    assert sorted(np.unique(ds.labels).tolist()) == [-1, 0, 1]


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
