import pytest
import pandas
import geopandas
import shapely
from nuc2seg.preprocessing import create_rasterized_dataset, tile_transcripts_to_disk
import numpy as np
import tempfile
import shutil
import glob
import os.path


@pytest.fixture(scope="function")
def test_nuclei_df():
    RECORDS = [
        {
            "geometry": shapely.Polygon(
                [(7.0, 7.0), (13.0, 7.0), (13.0, 13.0), (7.0, 13.0), (7.0, 7.0)]
            ),
            "nucleus_label": 1,
            "nucleus_centroid": shapely.Point(10.0, 10.0),
            "nucleus_centroid_x": 10.0,
            "nucleus_centroid_y": 10.0,
        },
        {
            "geometry": shapely.Polygon(
                [(17.0, 7.0), (23.0, 7.0), (23.0, 13.0), (17.0, 13.0), (17.0, 7.0)]
            ),
            "nucleus_label": 2,
            "nucleus_centroid": shapely.Point(20.0, 10.0),
            "nucleus_centroid_x": 20.0,
            "nucleus_centroid_y": 10.0,
        },
    ]

    return geopandas.GeoDataFrame(RECORDS).set_geometry("geometry")


@pytest.fixture(scope="function")
def test_transcripts_df():
    RECORDS = [
        # cell 1
        {
            "transcript_id": 1,
            "cell_id": "cell1",
            "overlaps_nucleus": 1,
            "feature_name": "gene1",
            "x_location": 10.0,
            "y_location": 11.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 0,
        },
        {
            "transcript_id": 2,
            "cell_id": "cell1",
            "overlaps_nucleus": 1,
            "feature_name": "gene1",
            "x_location": 12.0,
            "y_location": 11.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 0,
        },
        {
            "transcript_id": 3,
            "cell_id": "cell1",
            "overlaps_nucleus": 1,
            "feature_name": "gene1",
            "x_location": 11.0,
            "y_location": 10.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 0,
        },
        # cell 2
        {
            "transcript_id": 4,
            "cell_id": "cell2",
            "overlaps_nucleus": 1,
            "feature_name": "gene2",
            "x_location": 20.0,
            "y_location": 11.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 1,
        },
        {
            "transcript_id": 5,
            "cell_id": "cell1",
            "overlaps_nucleus": 1,
            "feature_name": "gene2",
            "x_location": 22.0,
            "y_location": 11.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 1,
        },
        {
            "transcript_id": 6,
            "cell_id": "cell1",
            "overlaps_nucleus": 1,
            "feature_name": "gene2",
            "x_location": 21.0,
            "y_location": 10.0,
            "z_location": 20.691404342651367,
            "qv": 40.0,
            "fov_name": "C10",
            "codeword_index": 28,
            "gene_id": 1,
        },
        # unlabeled transcripts
        {
            "transcript_id": 7,
            "cell_id": "UNASSIGNED",
            "overlaps_nucleus": 0,
            "feature_name": "gene1",
            "x_location": 10.0,
            "y_location": 5.0,
            "z_location": 13.079690933227539,
            "qv": 40.0,
            "fov_name": "C18",
            "codeword_index": 54,
            "gene_id": 0,
        },
        {
            "transcript_id": 8,
            "cell_id": "UNASSIGNED",
            "overlaps_nucleus": 0,
            "feature_name": "gene2",
            "x_location": 20.0,
            "y_location": 5.0,
            "z_location": 13.079690933227539,
            "qv": 40.0,
            "fov_name": "C18",
            "codeword_index": 54,
            "gene_id": 1,
        },
    ]

    df = pandas.DataFrame(RECORDS)

    return geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df["x_location"], df["y_location"]),
    )


def test_create_rasterized_dataset(test_nuclei_df, test_transcripts_df):
    np.random.seed(0)
    ds = create_rasterized_dataset(
        nuclei_geo_df=test_nuclei_df,
        tx_geo_df=test_transcripts_df,
        sample_area=shapely.Polygon([(1, 1), (30, 1), (30, 20), (1, 20), (1, 1)]),
        resolution=1,
        foreground_nucleus_distance=1,
        background_nucleus_distance=5,
        background_transcript_distance=3,
        background_pixel_transcripts=5,
    )

    assert ds.labels.shape == (30, 20)
    assert ds.transcripts.shape == (8, 3)
    assert ds.x_extent_pixels == 30
    assert ds.y_extent_pixels == 20
    assert ds.n_genes == 2

    # Assert coordinated are transformed relative to the bbox
    assert ds.transcripts[:, 0].min() == 9.0
    assert ds.transcripts[:, 1].min() == 4.0


def test_tile_transcripts_to_csv(test_transcripts_df):
    output_dir = tempfile.mkdtemp()

    try:
        tile_transcripts_to_disk(
            transcripts=test_transcripts_df,
            tile_size=(10, 10),
            overlap=0.5,
            output_dir=output_dir,
            output_format="csv",
        )
        output_fns = list(glob.glob(os.path.join(output_dir, "*.csv")))

        assert len(output_fns) == 8
        for fn in output_fns:
            df = pandas.read_csv(fn)
    finally:
        shutil.rmtree(output_dir)
