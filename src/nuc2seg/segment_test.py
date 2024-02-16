from nuc2seg.segment import (
    greedy_expansion,
    flow_destination,
    greedy_cell_segmentation,
    raster_to_polygon,
)
import numpy as np
import pytest


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
        result,
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
        ]
    )

    result = raster_to_polygon(arr)

    assert result
