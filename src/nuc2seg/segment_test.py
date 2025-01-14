import os.path
import shutil
import tempfile

import anndata
import geopandas
import numpy as np
import pytest
import torch
from blended_tiling import TilingModule
from shapely import Polygon, Point

from nuc2seg.data import ModelPredictions, Nuc2SegDataset
from nuc2seg.preprocessing import cart2pol
from nuc2seg.segment import (
    greedy_expansion,
    label_connected_components,
    flow_destination,
    greedy_cell_segmentation,
    raster_to_polygon,
    stitch_predictions,
    convert_segmentation_to_shapefile,
    collinear,
    fill_in_surrounded_unlabelled_pixels,
    update_labels_with_flow_values,
    greedy_expansion_step,
    cull_empty_pixels_from_segmentation,
    ray_tracing_cell_segmentation,
)
from nuc2seg.postprocess import convert_transcripts_to_anndata
from nuc2seg.utils import get_indices_for_ndarray


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

    for result in greedy_expansion(
        pixel_labels_arr.copy(),
        flow_labels.copy(),
        flow_labels2.copy(),
        flow_xy.copy(),
        flow_xy2.copy(),
        foreground_mask.copy(),
        max_expansion_steps=1,
    ):
        pass

    np.testing.assert_equal(result, np.array([[0, -1, 1, 1]]))

    for result in greedy_expansion(
        pixel_labels_arr.copy(),
        flow_labels.copy(),
        flow_labels2.copy(),
        flow_xy.copy(),
        flow_xy2.copy(),
        foreground_mask.copy(),
        max_expansion_steps=2,
    ):
        pass

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

    for result in greedy_expansion(
        pixel_labels_arr.copy(),
        flow_labels,
        flow_labels2,
        flow_xy,
        flow_xy2,
        foreground_mask,
        max_expansion_steps=1,
    ):
        pass

    np.testing.assert_equal(result, np.array([[0, -1, 1]]))


def test_greedy_cell_segmentation_early_stop():
    labels = np.zeros((64, 64))

    labels[22:42, 10:50] = -1
    labels[26:38, 22:42] = 1

    angles = np.zeros((64, 64))

    for x in range(64):
        for y in range(64):
            x_component = 32 - x
            y_component = 32 - y
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    ds = Nuc2SegDataset(
        labels=labels.astype(int),
        angles=angles,
        classes=np.ones((64, 64, 3)).astype(float),
        transcripts=np.array([[30, 11, 1], [30, 13, 0], [30, 20, 0], [30, 21, 0]]),
        bbox=np.array([0, 0, 64, 64]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    predictions = ModelPredictions(
        angles=angles,
        classes=np.ones((64, 64, 3)).astype(float),
        foreground=np.ones_like(labels).astype(float),
    )
    result = greedy_cell_segmentation(
        dataset=ds,
        predictions=predictions,
        prior_probs=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        expression_profiles=np.array(
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.98, 0.01, 0.01]]
        ),
        max_expansion_steps=15,
    )

    y, x = np.where(result.segmentation == 1)

    assert x.min() == 12


def test_ray_tracing_segmentation_early_stop():
    labels = np.zeros((64, 50))

    labels[22:42, 10:50] = -1
    labels[26:38, 22:42] = 1

    angles = np.zeros((64, 50))

    for x in range(64):
        for y in range(50):
            x_component = 32 - x
            y_component = 25 - y
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    ds = Nuc2SegDataset(
        labels=labels.astype(int),
        angles=angles,
        classes=np.ones((64, 50, 3)).astype(float),
        transcripts=np.array([[30, 11, 1], [30, 13, 0], [30, 20, 0], [30, 21, 0]]),
        bbox=np.array([0, 0, 64, 50]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    predictions = ModelPredictions(
        angles=angles,
        classes=np.ones((64, 50, 3)).astype(float),
        foreground=np.ones_like(labels).astype(float),
    )
    result = ray_tracing_cell_segmentation(
        dataset=ds,
        predictions=predictions,
        prior_probs=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        expression_profiles=np.array(
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.98, 0.01, 0.01]]
        ),
        max_length=15.0,
    )

    y, x = np.where(result.segmentation == 1)

    assert x.min() == 12


def test_greedy_expansion_step():
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

    greedy_expansion_step(
        pixel_labels_arr=pixel_labels_arr,
        indices_2d=start_xy,
        flow_labels=flow_labels,
        flow_labels2=flow_labels2,
        foreground_mask=foreground_mask,
        exclude_segments=np.array([1]),
    )
    # If excluded, segment 1 does not expand
    np.testing.assert_equal(pixel_labels_arr, np.array([[0, -1, -1, 1]]))

    greedy_expansion_step(
        pixel_labels_arr=pixel_labels_arr,
        indices_2d=start_xy,
        flow_labels=flow_labels,
        flow_labels2=flow_labels2,
        foreground_mask=foreground_mask,
        exclude_segments=np.array([2]),
    )
    # If not excluded, it does
    np.testing.assert_equal(pixel_labels_arr, np.array([[0, -1, 1, 1]]))


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

    assert output.classes.shape == (128, 128, 3)
    assert output.angles.shape == (128, 128)
    assert output.foreground.shape == (128, 128)


def test_convert_segmentation_to_shapefile():
    classes = np.zeros((64, 100, 4))

    classes[10:20, 10:20, :] = np.array([0.9, 0.01, 0.01, 0.01])
    classes[30:40, 30:40, :] = np.array([0.01, 0.9, 0.01, 0.01])

    predictions = ModelPredictions(
        angles=np.zeros((64, 100)),
        classes=classes,
        foreground=np.ones((64, 100)) * 0.5,
    )

    dataset = Nuc2SegDataset(
        labels=np.zeros((64, 100)),
        angles=np.zeros((64, 100)),
        classes=np.zeros((64, 100, 4)),
        transcripts=np.array([[0, 0, 0], [32, 32, 1], [35, 35, 2], [22, 22, 2]]),
        bbox=np.array([0, 0, 64, 100]),
        n_classes=3,
        n_genes=3,
        resolution=1,
    )

    segmentation = np.zeros((64, 100))

    segmentation[10:20, 10:20] = 1

    segmentation[30:40, 30:40] = 2
    segmentation[1, 1] = -1
    gdf = convert_segmentation_to_shapefile(
        dataset=dataset, predictions=predictions, segmentation=segmentation
    )

    assert gdf.shape[0] == 2
    assert gdf.iloc[0].unet_celltype_assignment == 0
    assert gdf.iloc[1].unet_celltype_assignment == 1
    assert gdf.iloc[0].geometry.area == 100
    assert gdf.iloc[1].geometry.area == 100
    assert gdf.iloc[0].unet_celltype_0_prob >= 0.89
    assert gdf.iloc[1].unet_celltype_1_prob >= 0.89


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


def test_label_free_greedy_expansion():
    x_extent_pixels = 3
    y_extent_pixels = 3
    angles = np.zeros((x_extent_pixels, y_extent_pixels))

    for x in range(x_extent_pixels):
        for y in range(y_extent_pixels):
            x_component = (x_extent_pixels / 2) - (x + 0.5)
            y_component = (y_extent_pixels / 2) - (y + 0.5)
            angle = cart2pol(x=x_component, y=y_component)
            angles[x, y] = angle[1]

    start_xy = np.mgrid[0:x_extent_pixels, 0:y_extent_pixels]
    start_xy = np.array(list(zip(start_xy[0].flatten(), start_xy[1].flatten())))

    flow_xy = flow_destination(start_xy, angles, np.sqrt(2))

    # Get the pixels that are sufficiently predicted to be foreground
    foreground_mask = np.ones((x_extent_pixels, y_extent_pixels)).astype(bool)[
        start_xy[:, 0], start_xy[:, 1]
    ]

    foreground_mask[0] = False

    result = label_connected_components(
        x_extent_pixels=x_extent_pixels,
        y_extent_pixels=y_extent_pixels,
        flow_xy=flow_xy,
        foreground_mask=foreground_mask,
        min_component_size=1,
    )

    np.testing.assert_equal(result, np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_fill_in_surrounded_unlabelled_pixels():
    labels = np.array(
        [
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ]
    )
    fill_in_surrounded_unlabelled_pixels(labels)
    assert np.all(labels == 1)

    labels = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    fill_in_surrounded_unlabelled_pixels(labels)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        ),
    )

    labels = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [1, 1, 1],
        ]
    )
    fill_in_surrounded_unlabelled_pixels(labels)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [1, 1, 1],
                [1, -1, -1],
                [1, 1, 1],
            ]
        ),
    )


def test_update_labels_with_flow_values():
    labels = np.array(
        [
            [1, 0, 0],
            [1, -1, -1],
            [0, 0, 0],
        ]
    )

    flow_labels = np.array(
        [
            [1, 0, 0],
            [1, 1, -1],
            [0, 0, 0],
        ]
    )

    indices_2d = get_indices_for_ndarray(labels.shape[0], labels.shape[1])
    x_indices = indices_2d[:, 0]
    y_indices = indices_2d[:, 1]

    flow_labels_flat = flow_labels[x_indices, y_indices]

    mask = labels == -1

    mask_flat = mask[x_indices, y_indices]

    update_labels_with_flow_values(
        labels=labels,
        update_mask=mask_flat,
        flow_labels_flat=flow_labels_flat,
        indices_2d=indices_2d,
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [1, 0, 0],
                [1, 1, -1],
                [0, 0, 0],
            ]
        ),
    )


def test_cull_empty_pixels_from_segmentation():
    segmentation = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )

    transcripts = np.array([[1, 1, 1], [2, 1, 2]])

    result = cull_empty_pixels_from_segmentation(segmentation, transcripts)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
            ]
        ),
    )
