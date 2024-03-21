import pandas

from nuc2seg.postprocess import stitch_shapes, read_baysor_results
from shapely import box
import geopandas as gpd
import pytest
import json
import tempfile
import os.path
import shutil


@pytest.fixture(scope="package")
def test_baysor_shapefile():
    geometries = [
        {
            "coordinates": [
                [
                    [6859.742, 2657.661],
                    [6860.28, 2657.3694],
                    [6859.5723, 2656.134],
                    [6858.534, 2655.866],
                    [6857.9883, 2654.3904],
                    [6857.6626, 2656.5325],
                    [6857.2764, 2656.554],
                    [6855.961, 2656.7205],
                    [6853.911, 2656.7864],
                    [6856.15, 2657.3735],
                    [6856.2163, 2657.4062],
                    [6857.623, 2657.0276],
                    [6858.165, 2657.1401],
                    [6858.2217, 2657.125],
                    [6859.742, 2657.661],
                ]
            ],
            "type": "Polygon",
            "cell": 7568,
        },
        {
            "coordinates": [
                [
                    [6648.5483, 3421.8179],
                    [6647.761, 3419.9482],
                    [6645.845, 3420.6394],
                    [6644.915, 3419.9849],
                    [6645.3506, 3420.5164],
                    [6645.333, 3422.3325],
                    [6645.297, 3424.1763],
                    [6646.4624, 3423.5347],
                    [6648.5483, 3421.8179],
                ]
            ],
            "type": "Polygon",
            "cell": 7834,
        },
    ]

    return {"geometries": geometries, "type": "FeatureCollection"}


@pytest.fixture(scope="package")
def test_baysor_output_table():
    records = [
        {
            "transcript_id": 281599530763812,
            "cell_id": 2,
            "overlaps_nucleus": 1,
            "gene": "SEC11C",
            "x": 6525.5605,
            "y": 2511.285,
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
            "x": 6527.4717,
            "y": 2722.4097,
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


def test_read_baysor_results(test_baysor_shapefile, test_baysor_output_table):
    tmpdir = tempfile.mkdtemp()
    geojson_fn = os.path.join(tmpdir, "test.geojson")
    csv_fn = os.path.join(tmpdir, "test.csv")
    with open(geojson_fn, "w") as f:
        json.dump(test_baysor_shapefile, f)

    test_baysor_output_table.to_csv(csv_fn, index=False)
    try:
        gdf = read_baysor_results(geojson_fn, csv_fn)

        assert len(gdf) == 2
        assert gdf["cell"].nunique() == 2
        assert gdf["cell"].iloc[0] == 7568
        assert gdf["cluster"].iloc[0] == 3
        assert gdf["cell"].iloc[1] == 7834
        assert gdf["cluster"].iloc[1] == 1
    finally:
        shutil.rmtree(tmpdir)


def test_stitch_shapes():
    upper_left_shape = gpd.GeoDataFrame({"geometry": [box(3, 3, 4, 4)]})
    bottom_right_shape = gpd.GeoDataFrame({"geometry": [box(18, 18, 19, 19)]})
    empty_shape = gpd.GeoDataFrame({"geometry": []})

    shapes = [
        upper_left_shape,
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        bottom_right_shape,
    ]
    result = stitch_shapes(shapes, (10, 10), (20, 20), 0.5)

    assert len(result) == 2

    shapes = [
        empty_shape.copy(),
        upper_left_shape,
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        empty_shape.copy(),
        bottom_right_shape,
    ]
    result = stitch_shapes(shapes, (10, 10), (20, 20), 0.5)
    assert len(result) == 1
