import shapely
import torch
import logging
import geopandas
import numpy as np
import anndata
import tqdm
import pandas as pd
import numpy_groupies as npg

from nuc2seg.preprocessing import pol2cart
from nuc2seg.data import (
    ModelPredictions,
    Nuc2SegDataset,
    SegmentationResults,
)
from scipy.sparse import csr_matrix
from shapely import Polygon, affinity, box
from blended_tiling import TilingModule

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


def greedy_expansion(
    start_xy,
    pixel_labels_arr,
    flow_labels,
    flow_labels2,
    flow_xy,
    flow_xy2,
    foreground_mask,
    max_expansion_steps=50,
):

    x_max = pixel_labels_arr.shape[0] - 1
    y_max = pixel_labels_arr.shape[1] - 1
    for _ in tqdm.trange(max_expansion_steps, desc="greedy_expansion", unit="step"):
        # Filter down to unassigned pixels that would flow to an assigned pixel
        pixel_labels_flat = pixel_labels_arr[start_xy[:, 0], start_xy[:, 1]]
        flow_labels_flat = flow_labels[start_xy[:, 0], start_xy[:, 1]]
        flow_labels_flat2 = flow_labels2[start_xy[:, 0], start_xy[:, 1]]
        update_mask = (
            foreground_mask
            & (pixel_labels_flat == -1)
            & ((flow_labels_flat != -1) | (flow_labels_flat2 != -1))
        )

        # If there are no pixels to update, just exit early
        if update_mask.sum() == 0:
            logger.info("No pixels left to update. Stopping expansion.")
            break

        # Update the filtered pixels to have the assignment of their flow neighbor
        # pixel_labels_arr[start_xy[update_mask,0], start_xy[update_mask,1]] = flow_labels_flat[update_mask]

        # Immediate neighbor flows
        flow1_mask = update_mask & (flow_labels_flat != -1)
        flow1_targets = flow_labels_flat[flow1_mask]
        flow1_connected = (
            (
                pixel_labels_arr[
                    (start_xy[flow1_mask, 0] - 1).clip(0, x_max),
                    (start_xy[flow1_mask, 1] - 1).clip(0, y_max),
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow1_mask, 0] + 1).clip(0, x_max),
                    (start_xy[flow1_mask, 1] - 1).clip(0, y_max),
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow1_mask, 0] - 1).clip(0, x_max),
                    (start_xy[flow1_mask, 1] + 1).clip(0, y_max),
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow1_mask, 0] + 1).clip(0, x_max),
                    (start_xy[flow1_mask, 1] + 1).clip(0, y_max),
                ]
                == flow1_targets
            )
        )
        flow1_targets[~flow1_connected] = -1
        pixel_labels_arr[start_xy[flow1_mask, 0], start_xy[flow1_mask, 1]] = (
            flow1_targets  # flow_labels_flat[flow1_mask]
        )

        # Slightly farther flows but still contiguous to the cell
        flow2_mask = update_mask & (flow_labels_flat == -1) & (flow_labels_flat2 != -1)
        flow2_targets = np.array(flow_labels_flat2[flow2_mask])
        flow2_connected = (
            (
                pixel_labels_arr[
                    (start_xy[flow2_mask, 0] - 1).clip(0, x_max),
                    (start_xy[flow2_mask, 1] - 1).clip(0, y_max),
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow2_mask, 0] + 1).clip(0, x_max),
                    (start_xy[flow2_mask, 1] - 1).clip(0, y_max),
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow2_mask, 0] - 1).clip(0, x_max),
                    (start_xy[flow2_mask, 1] + 1).clip(0, y_max),
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    (start_xy[flow2_mask, 0] + 1).clip(0, x_max),
                    (start_xy[flow2_mask, 1] + 1).clip(0, y_max),
                ]
                == flow2_targets
            )
        )
        flow2_targets[~flow2_connected] = -1
        pixel_labels_arr[start_xy[flow2_mask, 0], start_xy[flow2_mask, 1]] = (
            flow2_targets
        )

        # Post-process to remove island pixels surrounded by the same class on all sides
        island_mask = np.zeros(pixel_labels_arr.shape, dtype=bool)
        island_mask[1:-1, 1:-1] = (
            (pixel_labels_arr[1:-1, 1:-1] == -1)
            & (pixel_labels_arr[:-2, 1:-1] != -1)
            & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[2:, 1:-1])
            & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, :-2])
            & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, 2:])
        )
        pixel_labels_arr[island_mask] = pixel_labels_arr[:-2, 1:-1][
            island_mask[1:-1, 1:-1]
        ]

        # Update the flow labels
        flow_labels[start_xy[:, 0], start_xy[:, 1]] = np.maximum(
            pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
            pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
        )
        flow_labels2[start_xy[:, 0], start_xy[:, 1]] = np.maximum(
            pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
            pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
        )

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
):
    # Create a dataframe with an entry for every pixel
    pixel_labels_arr = dataset.labels.copy()

    # Determine where each pixel would flow to next
    start_xy = np.mgrid[0 : dataset.x_extent_pixels, 0 : dataset.y_extent_pixels]
    start_xy = np.array(list(zip(start_xy[0].flatten(), start_xy[1].flatten())))

    flow_xy = flow_destination(start_xy, predictions.angles, np.sqrt(2))
    flow_labels = (
        np.zeros((dataset.x_extent_pixels, dataset.y_extent_pixels), dtype=int) - 1
    )
    flow_labels[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy[:, 0], flow_xy[:, 1]
    ]

    flow_xy2 = flow_destination(start_xy, predictions.angles, np.sqrt(3))
    flow_labels2 = (
        np.zeros((dataset.x_extent_pixels, dataset.y_extent_pixels), dtype=int) - 1
    )
    flow_labels2[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy2[:, 0], flow_xy2[:, 1]
    ]

    # Get the pixels that are sufficiently predicted to be foreground
    foreground_mask = (predictions.foreground >= foreground_threshold)[
        start_xy[:, 0], start_xy[:, 1]
    ]

    flow_labels[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy[:, 0], flow_xy[:, 1]
    ]
    flow_labels2[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy2[:, 0], flow_xy2[:, 1]
    ]

    # Greedily expand the cell one pixel at a time
    result = greedy_expansion(
        start_xy,
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
        record["geometry"] = poly
        gb_idx = groupby_idx_lookup[cell_id]

        mean_probs = mean_class_prob_per_cell[gb_idx, :]
        class_assignment = int(np.argmax(mean_probs))

        record["class_assignment"] = class_assignment
        for i, val in enumerate(mean_probs):
            record[f"class_{i}_prob"] = val
        records.append(record)

    gdf = geopandas.GeoDataFrame(records, geometry="geometry")

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
    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_gdf, transcripts=transcript_gdf
    )
    sjoined_gdf.reset_index(inplace=True)

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
    )

    summed_counts_per_cell = pd.merge(
        summed_counts_per_cell,
        segmentation_gdf[["area", "centroid_x", "centroid_y"]],
        left_on="index",
        right_index=True,
    )

    del summed_counts_per_cell["index"]
    summed_counts_per_cell.reset_index(inplace=True)

    cell_u = list(sorted(summed_counts_per_cell.index.unique()))
    gene_u = list(sorted(summed_counts_per_cell[gene_name_column].unique()))

    summed_counts_per_cell["index"] = pd.Categorical(
        summed_counts_per_cell["index"], categories=cell_u, ordered=True
    )

    summed_counts_per_cell.set_index("index", inplace=True)

    data = summed_counts_per_cell["count"].tolist()

    summed_counts_per_cell["gene"] = pd.Categorical(
        summed_counts_per_cell[gene_name_column], categories=gene_u, ordered=True
    )
    row = summed_counts_per_cell.index.codes
    col = summed_counts_per_cell.gene.cat.codes

    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(cell_u), len(gene_u)))

    adata = anndata.AnnData(
        X=sparse_matrix,
        obsm={"spatial": summed_counts_per_cell[["centroid_x", "centroid_y"]].values},
        obs=summed_counts_per_cell[["area"]],
        var=pd.DataFrame(index=gene_u),
    )

    adata.obs_names.name = "cell_id"

    return adata
