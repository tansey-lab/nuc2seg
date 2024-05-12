import shapely
import torch
import logging
import geopandas
import numpy as np
import anndata
import tqdm
import pandas as pd
import numpy_groupies as npg

from shapely import affinity
from nuc2seg.preprocessing import pol2cart
from nuc2seg.celltyping import predict_celltype_probabilities_for_all_segments
from nuc2seg.data import (
    ModelPredictions,
    Nuc2SegDataset,
    SegmentationResults,
    CelltypingResults,
)
from typing import Callable, Optional
from nuc2seg.utils import get_indices_for_ndarray
from scipy.sparse import csr_matrix
from shapely import Polygon, affinity, box
from blended_tiling import TilingModule
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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

    return ModelPredictions(
        angles=angles_stitched.detach().cpu().numpy(),
        foreground=foreground_stitched.detach().cpu().numpy(),
        classes=class_stitched.detach().cpu().numpy(),
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
    pixel_labels_arr, indices_2d, flow_labels, flow_labels2, foreground_mask
):
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
    update_labels_with_flow_values(
        labels=pixel_labels_arr,
        update_mask=flow1_mask,
        flow_labels_flat=flow_labels_flat,
        indices_2d=indices_2d,
    )

    # Slightly farther flows but still contiguous to the cell
    flow2_mask = update_mask & (flow_labels_flat == -1) & (flow_labels_flat2 != -1)
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
):
    indices_2d = get_indices_for_ndarray(
        pixel_labels_arr.shape[0], pixel_labels_arr.shape[1]
    )

    for _ in tqdm.trange(max_expansion_steps, desc="greedy_expansion", unit="step"):
        # Expand one step
        greedy_expansion_step(
            pixel_labels_arr,
            indices_2d,
            flow_labels,
            flow_labels2,
            foreground_mask,
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

    return pixel_labels_arr


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
    foreground_threshold=0.5,
    max_expansion_steps=15,
    use_labels=True,
    min_component_size=20,
):
    indices_2d = get_indices_for_ndarray(
        dataset.x_extent_pixels, dataset.y_extent_pixels
    )  # <N pixel> x 2 array
    x_indices = indices_2d[:, 0]
    y_indices = indices_2d[:, 1]

    flow_xy = flow_destination(indices_2d, predictions.angles, np.sqrt(2))
    flow_xy2 = flow_destination(indices_2d, predictions.angles, np.sqrt(3))

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

    # Greedily expand the cell one pixel at a time
    result = greedy_expansion(
        pixel_labels_arr,
        flow_labels,
        flow_labels2,
        flow_xy,
        flow_xy2,
        foreground_mask,
        max_expansion_steps=max_expansion_steps,
    )
    return SegmentationResults(result)


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


def convert_segmentation_to_shapefile(
    segmentation, dataset: Nuc2SegDataset, predictions: ModelPredictions
):
    records = []
    classes = predictions.classes.transpose(1, 2, 0)
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
        poly = affinity.translate(
            poly,
            xoff=dataset.bbox[0],
            yoff=dataset.bbox[1],
        )

        record["geometry"] = poly
        gb_idx = groupby_idx_lookup[cell_id]

        mean_probs = mean_class_prob_per_cell[gb_idx, :]
        class_assignment = int(np.argmax(mean_probs))

        record["unet_celltype_assignment"] = class_assignment
        for i, val in enumerate(mean_probs):
            record[f"unet_celltype_{i}_prob"] = val
        records.append(record)

    gdf = geopandas.GeoDataFrame(records, geometry="geometry")

    gdf.reset_index(inplace=True, drop=True)
    gdf.reset_index(inplace=True, drop=False, names="segment_id")

    return gdf


def spatial_join_polygons_and_transcripts(
    boundaries: geopandas.GeoDataFrame, transcripts: geopandas.GeoDataFrame
):
    joined_gdf = geopandas.sjoin(boundaries, transcripts, how="inner")

    return joined_gdf


def convert_transcripts_to_anndata(
    transcript_gdf,
    segmentation_gdf,
    gene_name_column="feature_name",
    min_molecules_per_cell=None,
):
    segmentation_gdf["area"] = segmentation_gdf.geometry.area
    segmentation_gdf["centroid_x"] = segmentation_gdf.geometry.centroid.x
    segmentation_gdf["centroid_y"] = segmentation_gdf.geometry.centroid.y
    if "index" in transcript_gdf.columns:
        del transcript_gdf["index"]
    if "index" in segmentation_gdf.columns:
        del segmentation_gdf["index"]

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_gdf, transcripts=transcript_gdf
    )
    sjoined_gdf.reset_index(inplace=True, drop=False, names="index")

    before_dedupe = len(sjoined_gdf)

    # if more than one row has the same index_right value, drop until index_right is unique
    sjoined_gdf = sjoined_gdf.drop_duplicates(subset="index_right")

    after_dedupe = len(sjoined_gdf)

    logger.info(
        f"Dropped {before_dedupe - after_dedupe} transcripts assigned to multiple segments"
    )

    # filter transcripts mapped to cell where the total number of transcripts for that cell is less than
    # min_molecules_per_cell
    if min_molecules_per_cell is not None:
        before_min_molecules = len(sjoined_gdf)

        sjoined_gdf = sjoined_gdf.groupby("index").filter(
            lambda x: len(x) >= min_molecules_per_cell
        )

        after_min_molecules = len(sjoined_gdf)
        logger.info(
            f"Dropped {before_min_molecules - after_min_molecules} cells with fewer than {min_molecules_per_cell} transcripts"
        )

    summed_counts_per_cell = (
        sjoined_gdf.groupby(["index", gene_name_column])
        .size()
        .reset_index(name="count")
    ).rename(columns={"index": "cell_id"})

    cell_u = list(sorted(summed_counts_per_cell["cell_id"].unique()))
    gene_u = list(sorted(transcript_gdf[gene_name_column].unique()))

    summed_counts_per_cell["cell_id_idx"] = pd.Categorical(
        summed_counts_per_cell["cell_id"], categories=cell_u, ordered=True
    )

    summed_counts_per_cell[gene_name_column] = pd.Categorical(
        summed_counts_per_cell[gene_name_column], categories=gene_u, ordered=True
    )

    data = summed_counts_per_cell["count"].tolist()
    row = summed_counts_per_cell["cell_id_idx"].cat.codes
    col = summed_counts_per_cell[gene_name_column].cat.codes

    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(cell_u), len(gene_u)))

    shapefile_index = summed_counts_per_cell["cell_id"].unique()
    shapefile_index.sort()

    additional_columns = [x for x in segmentation_gdf.columns if x != "geometry"]

    adata = anndata.AnnData(
        X=sparse_matrix,
        obsm={
            "spatial": segmentation_gdf.loc[shapefile_index][
                ["centroid_x", "centroid_y"]
            ].values
        },
        obs=segmentation_gdf.loc[shapefile_index][additional_columns],
        var=pd.DataFrame(index=gene_u),
    )

    adata.obs_names.name = "cell_id"

    return adata
