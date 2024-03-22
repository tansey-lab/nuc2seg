import pandas

from nuc2seg.postprocess import (
    stitch_shapes,
    read_baysor_results,
)
from nuc2seg.segment import convert_transcripts_to_anndata
from shapely import box
import geopandas as gpd
import pytest
import json
import tempfile
import os.path
import shutil
import anndata


@pytest.fixture(scope="package")
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


@pytest.fixture(scope="package")
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


def test_read_baysor_results(test_baysor_shapefile, test_baysor_output_table):
    tmpdir = tempfile.mkdtemp()
    geojson_fn = os.path.join(tmpdir, "test.geojson")
    csv_fn = os.path.join(tmpdir, "test.csv")
    with open(geojson_fn, "w") as f:
        json.dump(test_baysor_shapefile, f)

    test_baysor_output_table.to_csv(csv_fn, index=False)
    try:
        shape_gdf, tx_gdf = read_baysor_results(geojson_fn, csv_fn)

        assert len(shape_gdf) == 2
        assert shape_gdf["cell"].nunique() == 2
        assert shape_gdf["cell"].iloc[0] == 7568
        assert shape_gdf["cluster"].iloc[0] == 3
        assert shape_gdf["cell"].iloc[1] == 7834
        assert shape_gdf["cluster"].iloc[1] == 1
        assert len(tx_gdf) == 2
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


def test_baysor_transcripts_to_anndata(test_baysor_shapefile, test_baysor_output_table):
    tmpdir = tempfile.mkdtemp()
    geojson_fn = os.path.join(tmpdir, "test.geojson")
    csv_fn = os.path.join(tmpdir, "test.csv")
    with open(geojson_fn, "w") as f:
        json.dump(test_baysor_shapefile, f)

    test_baysor_output_table.to_csv(csv_fn, index=False)
    try:
        shape_gdf, tx_gdf = read_baysor_results(geojson_fn, csv_fn)
        ad = convert_transcripts_to_anndata(
            transcript_gdf=tx_gdf,
            segmentation_gdf=shape_gdf,
            gene_name_column="gene",
        )
        ad.write_h5ad(os.path.join(tmpdir, "test.h5ad"))
        ad = anndata.read_h5ad(os.path.join(tmpdir, "test.h5ad"))

        assert ad.X.todense().shape == (2, 2)
        first_filter = ad.obsm["spatial"][:, 0] == 0.5
        second_filter = ad.obsm["spatial"][:, 0] == 10.5
        assert ad[ad.obs.index[first_filter], "SEC11C"].X.todense().item() == 1
        assert ad[ad.obs.index[first_filter], "LUM"].X.todense().item() == 0
        assert ad[ad.obs.index[second_filter], "LUM"].X.todense().item() == 1
        assert ad[ad.obs.index[second_filter], "SEC11C"].X.todense().item() == 0
    finally:
        shutil.rmtree(tmpdir)
