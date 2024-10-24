import pandas

from nuc2seg.postprocess import (
    stitch_shapes,
    read_baysor_shapes_with_cluster_assignment,
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


def test_read_baysor_shapes_with_cluster_assignment(
    test_baysor_shapefile, test_baysor_output_table
):
    tmpdir = tempfile.mkdtemp()
    geojson_fn = os.path.join(tmpdir, "test.geojson")
    csv_fn = os.path.join(tmpdir, "test.csv")
    with open(geojson_fn, "w") as f:
        json.dump(test_baysor_shapefile, f)

    test_baysor_output_table.to_csv(csv_fn, index=False)
    try:
        shape_gdf, tx_gdf = read_baysor_shapes_with_cluster_assignment(
            geojson_fn, csv_fn
        )

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
    upper_left_shape = gpd.GeoDataFrame({"cell": [1], "geometry": [box(3, 3, 4, 4)]})
    bottom_right_shape = gpd.GeoDataFrame(
        {"cell": [1], "geometry": [box(18, 18, 19, 19)]}
    )
    middle_shape = gpd.GeoDataFrame(
        {"cell": [1, 2], "geometry": [box(10, 10, 11, 11), box(12, 12, 13, 13)]}
    )
    empty_shape = gpd.GeoDataFrame({"geometry": [], "cell": []})

    shapes = [
        (
            0,
            upper_left_shape,
        ),
        (
            1,
            middle_shape,
        ),
        (
            2,
            middle_shape,
        ),
        (
            3,
            middle_shape,
        ),
        (
            4,
            middle_shape,
        ),
        (
            5,
            middle_shape,
        ),
        (
            6,
            middle_shape,
        ),
        (
            7,
            middle_shape,
        ),
        (
            8,
            bottom_right_shape,
        ),
    ]
    result = stitch_shapes(shapes, (10, 10), (20, 20), 0.5)

    assert len(result) == 4

    shapes = [
        (
            0,
            empty_shape.copy(),
        ),
        (
            1,
            upper_left_shape,
        ),
        (
            2,
            empty_shape.copy(),
        ),
        (
            3,
            empty_shape.copy(),
        ),
        (
            4,
            empty_shape.copy(),
        ),
        (
            5,
            empty_shape.copy(),
        ),
        (
            6,
            empty_shape.copy(),
        ),
        (
            7,
            empty_shape.copy(),
        ),
        (
            8,
            bottom_right_shape,
        ),
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
        shape_gdf, tx_gdf = read_baysor_shapes_with_cluster_assignment(
            geojson_fn, csv_fn
        )
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
