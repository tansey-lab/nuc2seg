import logging
from typing import Callable, Optional

import geopandas
import numpy as np
import numpy_groupies as npg
import pandas
import pandas as pd
import shapely
import torch
import tqdm
from blended_tiling import TilingModule
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import affinity, box

from nuc2seg.celltyping import predict_celltype_probabilities_for_all_segments
from nuc2seg.data import (
    ModelPredictions,
    Nuc2SegDataset,
    SegmentationResults,
)
from nuc2seg.preprocessing import pol2cart
from nuc2seg.utils import get_indices_for_ndarray, create_torch_polygon

logger = logging.getLogger(__name__)


def stitch_predictions(results, tiler: TilingModule):
    foreground = torch.sigmoid(results[..., 0])
    angles = torch.sigmoid(results[..., 1]) * 2 * torch.pi - torch.pi
    classes = torch.softmax(results[..., 2:], dim=-1)
    vector_x = 0.5 * torch.cos(angles)
    vector_y = 0.5 * torch.sin(angles)

    tile_mask = tiler.get_tile_masks()[:, 0, :, :]

    vector_x_tiles = vector_x * tile_mask
    vector_x_stitched = tiler.rebuild(vector_x_tiles[:, None, :, :]).squeeze()

    vector_y_tiles = vector_y * tile_mask
    vector_y_stitched = tiler.rebuild(vector_y_tiles[:, None, :, :]).squeeze()

    angles_stitched = torch.atan2(vector_y_stitched, vector_x_stitched)

    foreground_tiles = foreground * tile_mask
    foreground_stitched = tiler.rebuild(foreground_tiles[:, None, :, :]).squeeze()

    class_tiles = classes * tile_mask[..., None]
    class_stitched = tiler.rebuild(class_tiles.permute((0, 3, 1, 2))).squeeze()
    class_stitched = class_stitched.permute((1, 2, 0))

    return ModelPredictions(
        angles=angles_stitched.detach().cpu().numpy(),
        foreground=foreground_stitched.detach().cpu().numpy(),
        classes=class_stitched.detach().cpu().numpy(),
    )


def forward_pass_result_to_obj(value):
    foreground = torch.sigmoid(value[..., 0]).squeeze()
    angles = torch.sigmoid(value[..., 1]).squeeze() * 2 * torch.pi - torch.pi
    classes = torch.softmax(value[..., 2:], dim=-1).squeeze()
    vector_x = 0.5 * torch.cos(angles)
    vector_y = 0.5 * torch.sin(angles)
    angles_stitched = torch.atan2(vector_y, vector_x)

    return ModelPredictions(
        angles=angles_stitched.detach().cpu().numpy(),
        foreground=foreground.detach().cpu().numpy(),
        classes=classes.detach().cpu().numpy(),
    )


def update_labels_with_flow_values(
    labels: np.array,
    update_mask: np.array,
    flow_labels_flat: np.array,
    indices_2d: np.array,
):
    """
    :param labels: array of pixel labels, shape (n_width, n_height).
    :param update_mask: boolean mask of pixels for which we might potentially update the labels, shape (n_pixels,)
    :param flow_labels_flat: flow labels for each pixel, shape (n_pixels,)
    :param indices_2d: 2D indices for each pixel, shape (n_pixels, 2)

    :return: None, will update labels in-place
    """
    x_max = labels.shape[0] - 1
    y_max = labels.shape[1] - 1

    n_pixels_to_update = np.count_nonzero(update_mask)

    logger.debug(f"Possibly updating {n_pixels_to_update} pixels")

    flow_targets = flow_labels_flat[update_mask]

    # This will create a boolean mask of shape (n_pixels_to_update,) where each
    # pixel is True if the pixel that would be assigned a label based on the flow
    # has a diagonal neighbor with the same label.
    flow_is_connected = (
        (
            labels[
                (indices_2d[update_mask, 0] - 1).clip(0, x_max),
                (indices_2d[update_mask, 1] - 1).clip(0, y_max),
            ]
            == flow_targets
        )
        | (
            labels[
                (indices_2d[update_mask, 0] + 1).clip(0, x_max),
                (indices_2d[update_mask, 1] - 1).clip(0, y_max),
            ]
            == flow_targets
        )
        | (
            labels[
                (indices_2d[update_mask, 0] - 1).clip(0, x_max),
                (indices_2d[update_mask, 1] + 1).clip(0, y_max),
            ]
            == flow_targets
        )
        | (
            labels[
                (indices_2d[update_mask, 0] + 1).clip(0, x_max),
                (indices_2d[update_mask, 1] + 1).clip(0, y_max),
            ]
            == flow_targets
        )
    )
    flow_targets[~flow_is_connected] = -1
    labels[indices_2d[update_mask, 0], indices_2d[update_mask, 1]] = flow_targets


def fill_in_surrounded_unlabelled_pixels(
    labels: np.array,
):
    """
    :param labels: array of pixel labels, shape (n_width, n_height).
    """
    island_mask = np.zeros(labels.shape, dtype=bool)
    island_mask[1:-1, 1:-1] = (
        (labels[1:-1, 1:-1] == -1)
        & (labels[:-2, 1:-1] != -1)
        & (labels[:-2, 1:-1] == labels[2:, 1:-1])
        & (labels[:-2, 1:-1] == labels[1:-1, :-2])
        & (labels[:-2, 1:-1] == labels[1:-1, 2:])
    )
    labels[island_mask] = labels[:-2, 1:-1][island_mask[1:-1, 1:-1]]


def greedy_expansion_step(
    pixel_labels_arr,
    indices_2d,
    flow_labels,
    flow_labels2,
    foreground_mask,
    exclude_segments: Optional[np.array] = None,
):
    """
    :param pixel_labels_arr: array of pixel labels, shape (n_width, n_height).
    :param indices_2d: 2D indices for each pixel, shape (n_pixels, 2)
    :param flow_labels: flow labels for each pixel, shape (n_width, n_height)
    :param flow_labels2: flow labels for each pixel, shape (n_width, n_height)
    :param foreground_mask: boolean mask of pixels that are considered foreground, shape (n_width, n_height)
    :param exclude_segments: optional array of segment labels to exclude from the expansion
    :return: None, will update pixel_labels_arr in-place
    """
    x_indices = indices_2d[:, 0]
    y_indices = indices_2d[:, 1]

    # Filter down to unassigned pixels that would flow to an assigned pixel
    pixel_labels_flat = pixel_labels_arr[x_indices, y_indices]
    flow_labels_flat = flow_labels[x_indices, y_indices]
    flow_labels_flat2 = flow_labels2[x_indices, y_indices]
    update_mask = (
        foreground_mask
        & (pixel_labels_flat == -1)
        & ((flow_labels_flat != -1) | (flow_labels_flat2 != -1))
    )

    # Immediate neighbor flows
    flow1_mask = update_mask & (flow_labels_flat != -1)
    if exclude_segments is not None:
        flow1_mask = flow1_mask & ~np.isin(flow_labels_flat, exclude_segments)

    update_labels_with_flow_values(
        labels=pixel_labels_arr,
        update_mask=flow1_mask,
        flow_labels_flat=flow_labels_flat,
        indices_2d=indices_2d,
    )

    # Slightly farther flows but still contiguous to the cell
    flow2_mask = update_mask & (flow_labels_flat == -1) & (flow_labels_flat2 != -1)
    if exclude_segments is not None:
        flow2_mask = flow2_mask & ~np.isin(flow_labels_flat2, exclude_segments)

    update_labels_with_flow_values(
        labels=pixel_labels_arr,
        update_mask=flow2_mask,
        flow_labels_flat=flow_labels_flat,
        indices_2d=indices_2d,
    )

    # Post-process to remove island pixels surrounded by the same class on all sides
    fill_in_surrounded_unlabelled_pixels(pixel_labels_arr)


def greedy_expansion(
    pixel_labels_arr,
    flow_labels,
    flow_labels2,
    flow_xy,
    flow_xy2,
    foreground_mask,
    max_expansion_steps=50,
    exclude_segments_callback: Optional[Callable[[int, np.array], np.array]] = None,
    plotting_callback: Optional[Callable[[int, np.array], None]] = None,
):
    """
    :param pixel_labels_arr: array of pixel labels, shape (n_width, n_height).
    :param flow_labels: flow labels for each pixel, shape (n_width, n_height)
    :param flow_labels2: flow labels for each pixel, shape (n_width, n_height)
    :param flow_xy: flow destination for each pixel, shape (n_pixels, 2)
    :param flow_xy2: flow destination for each pixel, shape (n_pixels, 2)
    :param foreground_mask: boolean mask of pixels that are considered foreground, shape (n_width, n_height)
    :param max_expansion_steps: maximum number of expansion steps to take
    :param exclude_segments_callback: optional callback to determine which segments to exclude from expansion
    at a given iteration
    :param plotting_callback: optional callback to visualize the state of the segmentation at each iteration

    :return: Generator with the state of the labels at each iteration
    """
    indices_2d = get_indices_for_ndarray(
        pixel_labels_arr.shape[0], pixel_labels_arr.shape[1]
    )

    if plotting_callback is not None:
        plotting_callback(0, pixel_labels_arr.copy())

    for expansion_idx in tqdm.trange(
        max_expansion_steps, desc="greedy_expansion", unit="step"
    ):
        # Get segments to exclude
        if exclude_segments_callback is not None:
            exclude_segments = exclude_segments_callback(
                expansion_idx, pixel_labels_arr
            )
        else:
            exclude_segments = None

        # Expand one step
        greedy_expansion_step(
            pixel_labels_arr,
            indices_2d,
            flow_labels,
            flow_labels2,
            foreground_mask,
            exclude_segments=exclude_segments,
        )

        # Update the flow labels
        flow_labels[indices_2d[:, 0], indices_2d[:, 1]] = np.maximum(
            pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
            pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
        )
        flow_labels2[indices_2d[:, 0], indices_2d[:, 1]] = np.maximum(
            pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
            pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
        )

        if plotting_callback is not None:
            plotting_callback(expansion_idx + 1, pixel_labels_arr.copy())

        yield pixel_labels_arr


def label_connected_components(
    x_extent_pixels,
    y_extent_pixels,
    flow_xy,
    foreground_mask,
    min_component_size=20,
):
    start_xy = get_indices_for_ndarray(x_extent_pixels, y_extent_pixels)
    pixel_labels_arr = np.zeros((x_extent_pixels, y_extent_pixels), dtype=int)

    # Connect adjacent pixels where one pixel flows into the next
    rows = (start_xy[foreground_mask, 0] * y_extent_pixels) + start_xy[
        foreground_mask, 1
    ]
    cols = (flow_xy[foreground_mask, 0] * y_extent_pixels) + flow_xy[foreground_mask, 1]
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(foreground_mask.shape[0], foreground_mask.shape[0]),
    )

    # Estimate connected subgraphs
    n_connected, connected_labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    # Filter down to segments with enough pixels
    uniques, counts = np.unique(connected_labels[foreground_mask], return_counts=True)
    logger.info(f"Found {len(uniques)} connected components")

    to_keep = uniques[counts >= min_component_size]
    pixel_labels_arr[start_xy[foreground_mask, 0], start_xy[foreground_mask, 1]] = (
        connected_labels[foreground_mask]
    )
    pixel_labels_arr *= np.isin(pixel_labels_arr, to_keep).astype(int)

    _, inverse_indices = np.unique(pixel_labels_arr, return_inverse=True)
    pixel_labels_arr = np.reshape(inverse_indices, newshape=pixel_labels_arr.shape)

    logger.info(
        f"Found {len(to_keep)} connected components with at least {min_component_size} pixels"
    )

    # Post-process to remove island pixels surrounded by the same class on all sides
    island_mask = np.zeros(pixel_labels_arr.shape, dtype=bool)
    island_mask[1:-1, 1:-1] = (
        (pixel_labels_arr[1:-1, 1:-1] == 0)
        & (pixel_labels_arr[:-2, 1:-1] != 0)
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[2:, 1:-1])
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, :-2])
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, 2:])
    )
    pixel_labels_arr[island_mask] = pixel_labels_arr[:-2, 1:-1][island_mask[1:-1, 1:-1]]

    return pixel_labels_arr


def flow_destination(start_xy, angle_preds, magnitude):
    dxdy = np.array(pol2cart(magnitude, angle_preds))[
        :, start_xy[:, 0], start_xy[:, 1]
    ].T
    flow_xy = np.round(start_xy + dxdy).astype(int)
    flow_xy[:, 0] = flow_xy[:, 0].clip(0, angle_preds.shape[0] - 1).astype(int)
    flow_xy[:, 1] = flow_xy[:, 1].clip(0, angle_preds.shape[1] - 1).astype(int)
    return flow_xy


def greedy_cell_segmentation(
    dataset: Nuc2SegDataset,
    predictions: ModelPredictions,
    prior_probs: np.array,
    expression_profiles: np.array,
    foreground_threshold=0.5,
    max_expansion_steps=15,
    use_labels=True,
    min_component_size=20,
    flow1_magnitude=np.sqrt(2),
    flow2_magnitude=np.sqrt(3),
    plotting_callback: Optional[Callable[[int, np.array], None]] = None,
    use_early_stopping=True,
):
    indices_2d = get_indices_for_ndarray(
        dataset.x_extent_pixels, dataset.y_extent_pixels
    )  # <N pixel> x 2 array
    x_indices = indices_2d[:, 0]
    y_indices = indices_2d[:, 1]

    flow_xy = flow_destination(indices_2d, predictions.angles, flow1_magnitude)
    flow_xy2 = flow_destination(indices_2d, predictions.angles, flow2_magnitude)

    # Get the pixels that are sufficiently predicted to be foreground
    foreground_mask = (predictions.foreground >= foreground_threshold)[
        indices_2d[:, 0], indices_2d[:, 1]
    ]

    flow_labels = (
        np.zeros((dataset.x_extent_pixels, dataset.y_extent_pixels), dtype=int) - 1
    )
    flow_labels2 = (
        np.zeros((dataset.x_extent_pixels, dataset.y_extent_pixels), dtype=int) - 1
    )

    # Create a dataframe with an entry for every pixel
    if use_labels:
        pixel_labels_arr = dataset.labels.copy()
    else:
        pixel_labels_arr = label_connected_components(
            dataset.x_extent_pixels,
            dataset.y_extent_pixels,
            flow_xy,
            foreground_mask,
            min_component_size=min_component_size,
        )

    flow_labels[x_indices, y_indices] = pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]]
    flow_labels2[x_indices, y_indices] = pixel_labels_arr[
        flow_xy2[:, 0], flow_xy2[:, 1]
    ]

    result = pixel_labels_arr

    # Greedily expand the cell one pixel at a time, callback will record
    # the difference in celltype probabilities vector norm at each step
    # This metric becomes larger as the model becomes more confident in the celltype assignment
    gather_callback = GatherCelltypeProbabilitiesForSegments(
        prior_probs=prior_probs,
        expression_profiles=expression_profiles,
        transcripts=dataset.transcripts,
    )

    gather_callback(0, pixel_labels_arr)
    for idx, result in enumerate(
        greedy_expansion(
            pixel_labels_arr.copy(),
            flow_labels.copy(),
            flow_labels2.copy(),
            flow_xy.copy(),
            flow_xy2.copy(),
            foreground_mask.copy(),
            max_expansion_steps=max_expansion_steps,
        )
    ):
        gather_callback(idx + 1, result)

    if not use_early_stopping:
        return SegmentationResults(result)

    # Run the expansion again but this time stop each segments expansion at the point of
    # its maximum increase in celltype probability confidence
    exclude_callback = EnforceBestIterationForEachSegment(
        gather_callback.get_best_iteration_for_each_segment()
    )

    for result in greedy_expansion(
        pixel_labels_arr.copy(),
        flow_labels.copy(),
        flow_labels2.copy(),
        flow_xy.copy(),
        flow_xy2.copy(),
        foreground_mask.copy(),
        max_expansion_steps=max_expansion_steps,
        plotting_callback=plotting_callback,
        exclude_segments_callback=exclude_callback,
    ):
        pass
    return SegmentationResults(result)


class GatherCelltypeProbabilitiesForSegments:
    def __init__(self, transcripts, expression_profiles, prior_probs):
        self.norm_diff_per_step = []
        self.transcripts = transcripts
        self.expression_profiles = expression_profiles
        self.prior_probs = prior_probs
        self.original_probs = None

    def __call__(self, expansion_idx, pixel_labels_arr):
        probs = predict_celltype_probabilities_for_all_segments(
            labels=pixel_labels_arr,
            transcripts=self.transcripts,
            expression_profiles=self.expression_profiles,
            prior_probs=self.prior_probs,
        )  # N segment x N celltype

        if expansion_idx == 0:
            self.original_probs = probs

        norm_diff = norm((self.original_probs + probs) / 2.0, ord=2, axis=1) - norm(
            self.original_probs, ord=2, axis=1
        )
        self.norm_diff_per_step.append(norm_diff)
        return np.array([])

    def get_best_iteration_for_each_segment(self):
        arr = np.stack(self.norm_diff_per_step)

        # In case of ties take the highest iteration index
        return arr.shape[0] - arr[::-1, :].argmax(axis=0) - 1


class EnforceBestIterationForEachSegment:
    def __init__(self, best_iteration_for_each_segment):
        self.best_iteration_for_each_segment = best_iteration_for_each_segment

    def __call__(self, expansion_idx, pixel_labels_arr):
        return (
            np.arange(len(self.best_iteration_for_each_segment))[
                self.best_iteration_for_each_segment <= expansion_idx
            ]
            + 1
        )


def collinear(p1, p2, p3):
    return np.isclose(
        (p1[1] - p2[1]) * (p1[0] - p3[0]), (p1[1] - p3[1]) * (p1[0] - p2[0])
    )


def raster_to_polygon(raster):
    # get coordinates of true values in nparray
    x1, y1 = np.where(raster)
    x2 = x1 + 1
    y2 = y1 + 1

    shapes = []
    for x1, y1, x2, y2 in zip(x1, y1, x2, y2):
        shapes.append(box(x1, y1, x2, y2))

    return shapely.union_all(shapes)


def pixel_coords_to_polygon(coords):
    shapes = []
    for x1, y1 in coords:
        shapes.append(box(x1, y1, x1 + 1, y1 + 1))

    return shapely.union_all(shapes)


def segmentation_array_to_shapefile(segmentation):
    records = []
    segmentation_flattened = segmentation.flatten().astype(int)
    segmentation_flattened[segmentation_flattened == -1] = 0
    x, y = np.indices(segmentation.shape)

    # segmentation raster but as a dataframe with one row per pixel
    segmentation_df = pd.DataFrame(
        {"x": x.flatten(), "y": y.flatten(), "segmentation": segmentation.flatten()}
    )
    segmentation_df = segmentation_df[~segmentation_df["segmentation"].isin([-1, 0])]

    bag_of_coordinates_for_each_segment = segmentation_df.groupby("segmentation").apply(
        lambda x: list(zip(x["x"], x["y"]))
    )
    coordinates = bag_of_coordinates_for_each_segment.tolist()
    cell_ids = bag_of_coordinates_for_each_segment.index.tolist()

    for cell_id, coords in tqdm.tqdm(list(zip(cell_ids, coordinates))):
        record = {}
        poly = pixel_coords_to_polygon(coords)
        record["geometry"] = poly
        records.append(record)

    gdf = geopandas.GeoDataFrame(records, geometry="geometry")

    gdf.reset_index(inplace=True, drop=True)
    gdf.reset_index(inplace=True, drop=False, names="segment_id")

    return gdf


def cull_empty_pixels_from_segmentation(segmentation, transcripts, labels=None):
    x_extent, y_extent = segmentation.shape

    per_pixel_transcript_mask = np.zeros((x_extent, y_extent)).astype(bool)

    df = pandas.DataFrame(transcripts[:, :2], columns=["x", "y"])
    df = df.drop_duplicates()

    per_pixel_transcript_mask[df.x.values, df.y.values] = True

    modified = True
    while modified:
        modified = False
        for x_idx in range(x_extent):
            for y_idx in range(x_extent):
                if segmentation[x_idx, y_idx] <= 0:
                    continue

                surrounding_pixels = np.array(
                    [
                        [max(x_idx - 1, 0), y_idx],
                        [x_idx, max(y_idx - 1, 0)],
                        [x_idx, min(y_idx + 1, y_extent - 1)],
                        [min(x_idx + 1, x_extent - 1), y_idx],
                    ]
                )

                is_border = (
                    segmentation[
                        surrounding_pixels[:, 0], surrounding_pixels[:, 1]
                    ].min()
                    <= 0
                )

                if is_border and not per_pixel_transcript_mask[x_idx, y_idx]:
                    if (labels is not None) and labels[x_idx, y_idx] > 0:
                        continue
                    else:
                        segmentation[x_idx, y_idx] = 0
                    modified = True

    return segmentation


def convert_segmentation_to_shapefile(
    segmentation, dataset: Nuc2SegDataset, predictions: ModelPredictions, translate=True
):
    records = []
    classes = predictions.classes
    segmentation_flattened = segmentation.flatten().astype(int)
    segmentation_flattened[segmentation_flattened == -1] = 0
    x, y = np.indices(segmentation.shape)

    # segmentation raster but as a dataframe with one row per pixel
    segmentation_df = pd.DataFrame(
        {"x": x.flatten(), "y": y.flatten(), "segmentation": segmentation.flatten()}
    )

    segmentation_df = segmentation_df[~segmentation_df["segmentation"].isin([-1, 0])]

    if len(segmentation_df) == 0:
        return None

    bag_of_coordinates_for_each_segment = segmentation_df.groupby("segmentation").apply(
        lambda x: list(zip(x["x"], x["y"]))
    )
    coordinates = bag_of_coordinates_for_each_segment.tolist()
    cell_ids = bag_of_coordinates_for_each_segment.index.tolist()
    uniq = np.unique(segmentation_flattened).astype(int)
    groupby_idx_lookup = dict(zip(np.arange(len(uniq)), uniq))

    mean_class_prob_per_cell = np.zeros((len(uniq), classes.shape[2]))

    for i in range(classes.shape[2]):
        class_raveled = classes[:, :, i].ravel()
        mean_per_cell = npg.aggregate(
            segmentation_flattened, class_raveled, func="mean", fill_value=0
        )
        mean_class_prob_per_cell[:, i] = mean_per_cell

    for cell_id, coords in tqdm.tqdm(list(zip(cell_ids, coordinates))):
        record = {}
        poly = pixel_coords_to_polygon(coords)
        if translate:
            poly = affinity.translate(
                poly,
                xoff=dataset.bbox[0],
                yoff=dataset.bbox[1],
            )
        poly = affinity.scale(
            poly,
            xfact=dataset.resolution,
            yfact=dataset.resolution,
        )

        record["geometry"] = poly
        gb_idx = groupby_idx_lookup[cell_id]

        mean_probs = mean_class_prob_per_cell[gb_idx, :]
        class_assignment = int(np.argmax(mean_probs))

        record["unet_celltype_assignment"] = class_assignment
        for i, val in enumerate(mean_probs):
            record[f"unet_celltype_{i}_prob"] = val
        record["segment_id"] = cell_id - 1
        records.append(record)

    gdf = geopandas.GeoDataFrame(records, geometry="geometry")

    gdf.reset_index(inplace=True, drop=True)

    return gdf


def polygon_list_to_dense(polygons: list[shapely.Polygon], device):
    points = []
    dense_edge_vectors = []
    global_edge_index_to_polygon_map = {}
    offset = 0
    decomposed_polygons = []

    for poly_idx, poly in enumerate(polygons):
        if poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                decomposed_polygons.append((poly_idx, p))
        elif poly.geom_type == "Polygon":
            decomposed_polygons.append((poly_idx, poly))
        else:
            raise ValueError(f"Unsupported geom_type: {poly.geom_type}")

    for poly_idx, poly in decomposed_polygons:
        vertices = create_torch_polygon(poly, device)
        v = vertices.shape[0]

        for idx in range(v):
            global_edge_index_to_polygon_map[idx + offset] = poly_idx
        points.append(vertices)
        edge_vectors = vertices.roll(-1, dims=0) - vertices
        dense_edge_vectors.append(edge_vectors)
        offset += len(edge_vectors)

    dense_edge_vectors = torch.concatenate(dense_edge_vectors)
    points = torch.concatenate(points)
    return points, dense_edge_vectors, global_edge_index_to_polygon_map


def ray_polygon_intersection_2d(
    angles: torch.Tensor, polygons: list[shapely.Polygon], origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate ray-polygon intersections for a batch of angles in 2D.
    Each ray originates from its grid position (i, j).

    Args:
        angles: Tensor of shape (H, W) containing angles in radians
        polygon: List of shapely polygons to calculate intersection for
        origins: Tensor of shape (H, W) containing origin points for each angle

    Returns:
        intersect: Boolean tensor of shape (H, W) indicating which polygon each ray intersects (-1 if none)
        distances: Tensor of shape (H, W) containing distances to intersection points
                  (set to inf for non-intersecting rays)
    """
    W, H = angles.shape
    polygon_vertices, edge_vectors, global_edge_index_to_polygon_map = (
        polygon_list_to_dense(polygons=polygons, device=angles.device)
    )
    N = edge_vectors.shape[0]

    # Convert angles to direction vectors (normalized)
    directions = torch.stack(
        [torch.cos(angles), torch.sin(angles)], dim=-1
    )  # (W, H, 2)

    # Expand dimensions for broadcasting
    # Ray directions: (H, W, 1, 2) for broadcasting with N edges
    # Origins: (H, W, 1, 2) for broadcasting with N edges
    # Polygon points: (1, 1, N, 2) for broadcasting with H×W rays
    # Edges: (1, 1, N, 2) for broadcasting with H×W rays
    directions = directions.unsqueeze(2)  # (H, W, 1, 2)
    origins = origins.unsqueeze(2)  # (H, W, 1, 2)
    polygon_points = polygon_vertices.view(1, 1, N, 2).expand(W, H, N, 2)
    edges = edge_vectors.view(1, 1, N, 2).expand(W, H, N, 2)

    # Calculate determinant (cross product in 2D)
    # If det is 0, ray is parallel to edge
    det = directions[..., 0] * edges[..., 1] - directions[..., 1] * edges[..., 0]

    # Calculate relative vector from ray origin to polygon points
    rel_vec = polygon_points - origins

    # Calculate intersection parameters
    # t: distance along ray
    # u: interpolation factor along edge (0 to 1)
    t = (rel_vec[..., 0] * edges[..., 1] - rel_vec[..., 1] * edges[..., 0]) / (
        det + 1e-10
    )
    u = (
        rel_vec[..., 0] * directions[..., 1] - rel_vec[..., 1] * directions[..., 0]
    ) / (det + 1e-10)

    # Check valid intersections:
    # 1. Ray and edge aren't parallel (det != 0)
    # 2. Intersection point is along the edge (0 <= u <= 1)
    # 3. Intersection point is in front of ray origin (t > 0)
    valid = (torch.abs(det) > 1e-10) & (u >= 0) & (u <= 1) & (t >= 0)

    # Get minimum distance for each ray
    distances = torch.where(valid, t, torch.inf)
    min_distances, _ = torch.min(distances, dim=-1)
    # Final intersection check (at least one valid intersection)
    intersect = torch.any(valid, dim=-1)

    # Get the id of each polygon that each ray is intersecting
    face_indices = distances.argmin(dim=-1)
    face_indices[~intersect] = -1
    transform = np.vectorize(lambda x: global_edge_index_to_polygon_map.get(x, -1))
    polygon_indices = transform(face_indices.cpu().numpy())

    return intersect, torch.tensor(polygon_indices), min_distances


def ray_tracing_cell_segmentation(
    dataset: Nuc2SegDataset,
    predictions: ModelPredictions,
    prior_probs: np.array,
    expression_profiles: np.array,
    device: str = "cpu",
    foreground_threshold: float = 0.5,
    max_length: float = 15,
    use_labels=True,
    use_early_stopping=True,
):
    angles = torch.tensor(predictions.angles, device=device)
    polygons: list[shapely.Polygon] = segmentation_array_to_shapefile(
        dataset.labels
    ).geometry.to_list()

    i, j = torch.meshgrid(
        torch.arange(angles.shape[0]), torch.arange(angles.shape[1]), indexing="ij"
    )
    origins = torch.stack([i.float(), j.float()], dim=-1) + 0.5
    intersect_mask, polygon_intersected, min_distances = ray_polygon_intersection_2d(
        angles, polygons, origins
    )

    # fix indexing of segments to match assumptions
    polygon_intersected = polygon_intersected.cpu().numpy()
    polygon_intersected[(polygon_intersected > -1)] += 1
    polygon_intersected = np.maximum(polygon_intersected, dataset.labels)
    min_distances = min_distances.cpu().numpy()
    min_distances[dataset.labels > 0] = 0

    gather_callback = GatherCelltypeProbabilitiesForSegments(
        prior_probs=prior_probs,
        expression_profiles=expression_profiles,
        transcripts=dataset.transcripts,
    )

    ray_steps = np.linspace(
        0, max_length, int(np.ceil(max_length / dataset.resolution))
    )

    for idx, max_ray_length in enumerate(ray_steps):
        step_segmentation = polygon_intersected.copy()
        step_segmentation[~intersect_mask] = 0
        step_segmentation[(min_distances > max_ray_length)] = 0

        gather_callback(idx, step_segmentation)

    if not use_early_stopping:
        return SegmentationResults(step_segmentation)

    best_iteration_per_segment = gather_callback.get_best_iteration_for_each_segment()

    result = np.zeros_like(predictions.angles)

    for segment_index, best_iteration in enumerate(best_iteration_per_segment):
        max_ray_length = ray_steps[best_iteration]
        segment_index = segment_index + 1

        mask = polygon_intersected == segment_index
        mask &= min_distances <= max_ray_length
        mask &= predictions.foreground >= foreground_threshold
        mask &= dataset.labels != 0
        mask |= dataset.labels == segment_index
        result[mask] = segment_index

    return SegmentationResults(result)
