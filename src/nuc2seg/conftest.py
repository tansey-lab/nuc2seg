import pytest
import pandas
import geopandas
import shapely


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def test_baysor_shapefile():
    geometries = [
        {
            "coordinates": [
                [
                    [0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                ]
            ],
            "type": "Polygon",
            "cell": 7568,
        },
        {
            "coordinates": [
                [
                    [10, 10],
                    [10, 11],
                    [11, 11],
                    [11, 10],
                ]
            ],
            "type": "Polygon",
            "cell": 7834,
        },
        {
            "coordinates": [
                [
                    [77, 77],
                    [88, 88],
                ]
            ],
            "type": "Polygon",
            "cell": 7834,
        },
    ]

    return {"geometries": geometries, "type": "FeatureCollection"}


@pytest.fixture(scope="session")
def test_baysor_output_table():
    records = [
        {
            "transcript_id": 281599530763812,
            "cell_id": 2,
            "overlaps_nucleus": 1,
            "gene": "SEC11C",
            "x": 0.5,
            "y": 0.5,
            "z": 34.055805,
            "qv": 21.204987,
            "gene_id": 4,
            "nucleus_id": 139210,
            "molecule_id": 2,
            "prior_segmentation": 784,
            "confidence": 0.99996,
            "cluster": 3,
            "cell": "CRb5afb8686-7568",
            "assignment_confidence": 1.0,
            "is_noise": False,
            "ncv_color": "#9DCDBB",
        },
        {
            "transcript_id": 281599530763828,
            "cell_id": 7729,
            "overlaps_nucleus": 0,
            "gene": "LUM",
            "x": 10.5,
            "y": 10.5,
            "z": 36.20927,
            "qv": 40.0,
            "gene_id": 11,
            "nucleus_id": 0,
            "molecule_id": 3,
            "prior_segmentation": 0,
            "confidence": 0.68609,
            "cluster": 1,
            "cell": "CRb5afb8686-7834",
            "assignment_confidence": 0.98,
            "is_noise": False,
            "ncv_color": "#003262",
        },
    ]
    return pandas.DataFrame(records)


@pytest.fixture(scope="session")
def test_nucleus_boundaries():
    return pandas.DataFrame(
        [
            {
                "cell_id": 7568,
                "vertex_x": 7.0,
                "vertex_y": 7.0,
            },
            {
                "cell_id": 7568,
                "vertex_x": 13.0,
                "vertex_y": 7.0,
            },
            {
                "cell_id": 7568,
                "vertex_x": 13.0,
                "vertex_y": 13.0,
            },
            {
                "cell_id": 7568,
                "vertex_x": 7.0,
                "vertex_y": 13.0,
            },
            {
                "cell_id": 7834,
                "vertex_x": 17.0,
                "vertex_y": 7.0,
            },
            {
                "cell_id": 7834,
                "vertex_x": 23.0,
                "vertex_y": 7.0,
            },
            {
                "cell_id": 7834,
                "vertex_x": 23.0,
                "vertex_y": 13.0,
            },
            {
                "cell_id": 7834,
                "vertex_x": 17.0,
                "vertex_y": 13.0,
            },
        ]
    )
