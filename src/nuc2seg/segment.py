import torch
import tqdm
import logging
import cv2
import geopandas
import numpy as np
import anndata
import tqdm
import pandas as pd

from nuc2seg.preprocessing import pol2cart
from nuc2seg.data import (
    ModelPredictions,
    Nuc2SegDataset,
    SegmentationResults,
)
from scipy.sparse import csr_matrix
from shapely import Polygon, affinity
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
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(
        raster.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize lists to hold outer contours and holes
    outer_contours = []
    holes = []

    for i, contour in enumerate(contours):
        # The contour points are in (row, column) format, so we flip them to (x, y)
        coords = contour.squeeze(axis=1)
        xy = [(p[1], p[0]) for p in coords]

        # Check if this contour has a parent
        if hierarchy[0][i][3] == -1:
            # No parent, so it's an external contour
            outer_contours.append(xy)
        else:
            # Has a parent, so it's a hole
            holes.append(xy)

    # Assuming single polygon, we take the first outer contour
    # You can loop over outer_contours to create multiple polygons if needed
    if outer_contours:
        exterior = outer_contours[0]

        # remove duplicate points without changing order
        seen = []
        for pt in exterior:
            if pt in seen:
                continue
            # detect if last 3 points are collinear
            if len(seen) > 2 and collinear(seen[-2], seen[-1], pt):
                continue
            else:
                seen.append(pt)

        return Polygon(seen, holes=[])


def convert_segmentation_to_shapefile(
    segmentation, dataset: Nuc2SegDataset, predictions: ModelPredictions
):
    x1, y1, x2, y2 = dataset.bbox

    nuclei_id = np.setdiff1d(np.unique(segmentation), [-1, 0])

    gdf = geopandas.GeoDataFrame(
        index=nuclei_id,
    )

    for value in tqdm.tqdm(np.unique(segmentation)):
        if value in [-1, 0]:
            continue
        mask = segmentation == value
        try:
            poly = raster_to_polygon(mask)
        except ValueError:
            logger.exception(
                "Failed to convert segmentation to poly (probably too small)"
            )
            continue

        translated_poly = affinity.translate(poly, xoff=x1, yoff=y1)

        gdf.loc[value, "geometry"] = translated_poly

        mean_probs = predictions.classes.transpose(1, 2, 0)[mask, :].mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()
        class_assignment = int(np.argmax(mean_probs))

        gdf.loc[value, "class_assignment"] = class_assignment
        for i, val in enumerate(mean_probs):
            gdf.loc[value, f"class_{i}_prob"] = val

    gdf.set_geometry("geometry", inplace=True)

    return gdf


def spatial_join_polygons_and_transcripts(
    boundaries: geopandas.GeoDataFrame, transcripts: geopandas.GeoDataFrame
):
    joined_gdf = geopandas.sjoin(boundaries, transcripts, how="inner")

    return joined_gdf


def convert_transcripts_to_anndata(
    transcript_gdf, segmentation_gdf, gene_name_column="feature_name"
):
    segmentation_gdf["area"] = segmentation_gdf.geometry.area
    segmentation_gdf["centroid_x"] = segmentation_gdf.geometry.centroid.x
    segmentation_gdf["centroid_y"] = segmentation_gdf.geometry.centroid.y
    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_gdf, transcripts=transcript_gdf
    )

    cell_u = list(sorted(sjoined_gdf.index_right.unique()))
    gene_u = list(sorted(sjoined_gdf[gene_name_column].unique()))

    sjoined_gdf["index_right"] = pd.Categorical(
        sjoined_gdf["index_right"], categories=cell_u, ordered=True
    )

    sjoined_gdf.set_index("index_right", inplace=True)

    sjoined_gdf["count"] = 1

    data = sjoined_gdf["count"].tolist()

    sjoined_gdf["gene"] = pd.Categorical(
        sjoined_gdf[gene_name_column], categories=gene_u, ordered=True
    )
    row = sjoined_gdf.index.codes
    col = sjoined_gdf.gene.cat.codes

    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(cell_u), len(gene_u)))

    adata = anndata.AnnData(
        X=sparse_matrix,
        obsm={"spatial": sjoined_gdf[["centroid_x", "centroid_y"]].values},
        obs=sjoined_gdf[["area"]],
        var=pd.DataFrame(index=gene_u),
    )

    adata.obs_names.name = "cell_id"

    return adata
