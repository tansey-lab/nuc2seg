from nuc2seg.postprocess import stitch_shapes
from shapely import box
import geopandas as gpd


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
