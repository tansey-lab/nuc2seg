from nuc2seg.segment import (
    greedy_expansion,
    flow_destination,
    greedy_cell_segmentation,
    raster_to_polygon,
    stitch_predictions,
    convert_segmentation_to_shapefile,
    convert_transcripts_to_anndata,
    collinear,
)
import numpy as np
import pytest
import torch
import geopandas
import shutil
import os.path
import tempfile
import anndata

from shapely import Polygon, Point
from blended_tiling import TilingModule
from nuc2seg.data import ModelPredictions, Nuc2SegDataset


def test_greedy_expansion_updates_pixel_with_distance_according_to_iter():
    pixel_labels_arr = np.array(
        [
            [0, -1, -1, 1],
        ]
    )

    flow_labels = np.array(
        [
            [0, -1, 1, 1],
        ]
    )

    flow_labels2 = np.array(
        [
            [0, -1, 1, 1],
        ]
    )

    start_xy = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])

    foreground_mask = (pixel_labels_arr != 0)[start_xy[:, 0], start_xy[:, 1]]

    flow_xy = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 3],
        ]
    )

    flow_xy2 = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 3],
        ]
    )

    result = greedy_expansion(
        start_xy.copy(),
        pixel_labels_arr.copy(),
        flow_labels.copy(),
        flow_labels2.copy(),
        flow_xy.copy(),
        flow_xy2.copy(),
        foreground_mask.copy(),
        max_expansion_steps=1,
    )

    np.testing.assert_equal(result, np.array([[0, -1, 1, 1]]))

    result = greedy_expansion(
        start_xy.copy(),
        pixel_labels_arr.copy(),
        flow_labels.copy(),
        flow_labels2.copy(),
        flow_xy.copy(),
        flow_xy2.copy(),
        foreground_mask.copy(),
        max_expansion_steps=2,
    )

    np.testing.assert_equal(result, np.array([[0, 1, 1, 1]]))


def test_greedy_expansion_doesnt_update_pixel():
    pixel_labels_arr = np.array(
        [
            [0, -1, 1],
        ]
    )

    flow_labels = np.array(
        [
            [0, -1, 1],
        ]
    )

    flow_labels2 = np.array(
        [
            [0, -1, 1],
        ]
    )

    start_xy = np.array([[0, 0], [0, 1], [0, 2]])

    foreground_mask = (pixel_labels_arr != 0)[start_xy[:, 0], start_xy[:, 1]]

    flow_xy = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
        ]
    )

    flow_xy2 = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
        ]
    )

    result = greedy_expansion(
        start_xy,
        pixel_labels_arr.copy(),
        flow_labels,
        flow_labels2,
        flow_xy,
        flow_xy2,
        foreground_mask,
        max_expansion_steps=1,
    )

    np.testing.assert_equal(result, np.array([[0, -1, 1]]))


@pytest.mark.parametrize(
    "angle,expected_destination",
    [
        (-np.pi, [0, 1]),
        (-3 * np.pi / 4, [0, 0]),
        (-np.pi / 2, [1, 0]),
        (-np.pi / 4, [2, 0]),
        (0, [2, 1]),
        (np.pi / 4, [2, 2]),
        (np.pi / 2, [1, 2]),
        (3 * np.pi / 4, [0, 2]),
        (np.pi, [0, 1]),
    ],
)
def test_flow_destination(angle, expected_destination):
    start_xy = np.array([[1, 1]])

    angles = np.array([[0, 0, 0], [0, angle, 0], [0, 0, 0]])

    result = flow_destination(start_xy, angles, np.sqrt(2))

    np.testing.assert_equal(result, np.array([expected_destination]))


def test_greedy_cell_segmentation(mocker):
    mock_dataset = mocker.Mock()
    mock_dataset.labels = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [1, 1, 1],
        ]
    )
    mock_dataset.x_extent_pixels = 3
    mock_dataset.y_extent_pixels = 3

    mock_predictions = mocker.Mock()

    mock_predictions.angles = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    mock_predictions.foreground = np.array(
        [
            [0.1, 0.99, 0.1],
            [0.1, 0.99, 0.1],
            [0.1, 0.99, 0.1],
        ]
    )

    result = greedy_cell_segmentation(
        mock_dataset, mock_predictions, foreground_threshold=0.5, max_expansion_steps=10
    )

    np.testing.assert_equal(
        result.segmentation,
        np.array(
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
            ]
        ),
    )


def test_raster_to_polygon():
    arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )

    result = raster_to_polygon(arr)

    assert result.area == 9
    assert result.bounds == (1, 1, 4, 4)


def test_non_convex_raster_to_polygon():
    # draw a serif letter S as a numpy array in pixels
    arr = np.zeros((20, 20))
    arr[1:3, 1:19] = 1
    arr[3:10, 1:3] = 1
    arr[10:18, 17:19] = 1
    arr[18:20, 1:19] = 1
    arr[9, 1:19] = 1
    arr[3, 18] = 1
    arr[17, 1] = 1
    result = raster_to_polygon(arr)

    assert result.area == 120


def test_stitch_predictions():
    pred_result = torch.ones((9, 64, 64, 5))
    tiling_module = TilingModule(
        base_size=(128, 128),
        tile_size=(64, 64),
        tile_overlap=(0.25, 0.25),
    )

    output = stitch_predictions(results=pred_result, tiler=tiling_module)

    assert output.classes.shape == (3, 128, 128)
    assert output.angles.shape == (128, 128)
    assert output.foreground.shape == (128, 128)


def test_convert_segmentation_to_shapefile():
    classes = np.zeros((64, 64, 4))

    classes[10:20, 10:20, :] = np.array([0.9, 0.01, 0.01, 0.01])
    classes[30:40, 30:40, :] = np.array([0.01, 0.9, 0.01, 0.01])
    classes = classes.transpose((2, 0, 1))

    predictions = ModelPredictions(
        angles=np.zeros((64, 64)),
        classes=classes,
        foreground=np.ones((64, 64)) * 0.5,
    )

    dataset = Nuc2SegDataset(
        labels=np.zeros((64, 64)),
        angles=np.zeros((64, 64)),
        classes=np.zeros((64, 64, 4)),
        transcripts=np.array([[0, 0, 0], [32, 32, 1], [35, 35, 2], [22, 22, 2]]),
        bbox=np.array([0, 0, 64, 64]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    segmentation = np.zeros((64, 64))

    segmentation[10:20, 10:20] = 1

    segmentation[30:40, 30:40] = 2

    gdf = convert_segmentation_to_shapefile(
        dataset=dataset, predictions=predictions, segmentation=segmentation
    )

    assert gdf.shape[0] == 2
    assert gdf.iloc[0].class_assignment == 0
    assert gdf.iloc[1].class_assignment == 1
    assert gdf.iloc[0].geometry.area == 100
    assert gdf.iloc[1].geometry.area == 100


def test_convert_transcripts_to_anndata():
    tmpdir = tempfile.mkdtemp()
    boundaries = geopandas.GeoDataFrame(
        [
            [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
            [Polygon([(0, 0), (0, 1), (3, 1), (3, 0)])],
        ],
        columns=["geometry"],
    )

    transcripts = geopandas.GeoDataFrame(
        [
            ["a", Point(0.5, 0.5)],
            ["a", Point(0.5, 0.5)],
            ["b", Point(2, 0.5)],
        ],
        columns=["feature_name", "geometry"],
    )

    try:
        adata = convert_transcripts_to_anndata(
            transcript_gdf=transcripts, segmentation_gdf=boundaries
        )
        adata.write_h5ad(os.path.join(tmpdir, "test.h5ad"))
        adata = anndata.read_h5ad(os.path.join(tmpdir, "test.h5ad"))

        assert adata.n_vars == 2
        assert adata.n_obs == 2
        np.testing.assert_array_equal(
            adata.obsm["spatial"], np.array([[0.5, 0.5], [1.5, 0.5]])
        )

        results = adata.X.todense()

        assert results.sum() == 3

        assert adata.var_names.tolist() == ["a", "b"]

        adata = convert_transcripts_to_anndata(
            transcript_gdf=transcripts,
            segmentation_gdf=boundaries,
            min_molecules_per_cell=10,
        )

        assert adata.n_obs == 0
    finally:
        shutil.rmtree(tmpdir)


def test_collinear():
    assert collinear((0, 0), (1, 1), (2, 2))
    assert collinear((0, 0), (1, 0), (2, 0))
    assert not collinear((0, 0), (1, 0), (2, 1))
    assert not collinear((0, 0), (1, 1), (2, 0))
    assert not collinear((0, 1), (1, 1), (2, 0))
