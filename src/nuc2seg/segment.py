import torch
import numpy as np
import tqdm
import logging
from scipy.special import expit, softmax
from nuc2seg.preprocessing import pol2cart
from nuc2seg.data import collate_tiles, ModelPredictions, Nuc2SegDataset

logger = logging.getLogger(__name__)


def stitch_predictions(model, dataloader):
    model.eval()
    foreground_list = []
    class_list = []

    vector_x_list = []
    vector_y_list = []
    for batch in tqdm.tqdm(dataloader, desc="Stitching predictions", unit="batch"):
        tile = collate_tiles([batch])

        x, y, z, labels, angles, classes, label_mask, nucleus_mask, location = (
            tile["X"],
            tile["Y"],
            tile["gene"],
            tile["labels"].numpy().copy().astype(int),
            tile["angles"].numpy().copy().astype(float),
            tile["classes"].numpy().copy().astype(int),
            tile["label_mask"].numpy().copy().astype(bool),
            tile["nucleus_mask"].numpy().copy().astype(bool),
            tile["location"],
        )
        mask_pred = model(x, y, z).detach().numpy().copy()
        foreground_pred = expit(mask_pred[..., 0])
        angles_pred = expit(mask_pred[..., 1]) * 2 * np.pi - np.pi
        class_pred = softmax(mask_pred[..., 2:], axis=-1)
        vector_x_list.append(0.5 * np.cos(angles_pred.squeeze()))
        vector_y_list.append(0.5 * np.sin(angles_pred.squeeze()))
        foreground_list.append(foreground_pred.squeeze())
        class_list.append(class_pred.squeeze())

    all_vector_x = torch.tensor(np.stack(vector_x_list, axis=0))
    all_vector_y = torch.tensor(np.stack(vector_y_list, axis=0))

    all_foreground = torch.tensor(np.stack(foreground_list, axis=0))
    all_classes = torch.tensor(np.stack(class_list, axis=0))

    tile_mask = dataloader.tiler.get_tile_masks()[:, 0, :, :]

    vector_x_tiles = all_vector_x * tile_mask
    vector_x_stitched = dataloader.tiler.rebuild(
        vector_x_tiles[:, None, :, :]
    ).squeeze()

    vector_y_tiles = all_vector_y * tile_mask
    vector_y_stitched = dataloader.tiler.rebuild(
        vector_y_tiles[:, None, :, :]
    ).squeeze()

    angles_stitched = torch.atan2(vector_y_stitched, vector_x_stitched)

    foreground_tiles = all_foreground * tile_mask
    foreground_stitched = dataloader.tiler.rebuild(
        foreground_tiles[:, None, :, :]
    ).squeeze()

    class_tiles = all_classes * tile_mask[..., None]
    class_stitched = dataloader.tiler.rebuild(
        class_tiles.permute((0, 3, 1, 2))
    ).squeeze()

    return ModelPredictions(
        angles=angles_stitched.detach().numpy(),
        foreground=foreground_stitched.detach().numpy(),
        classes=class_stitched.detach().numpy(),
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
        flow1_targets = np.array(flow_labels_flat[flow1_mask])
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
                    start_xy[flow2_mask, 0] - 1, start_xy[flow2_mask, 1] - 1
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow2_mask, 0] + 1, start_xy[flow2_mask, 1] - 1
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow2_mask, 0] - 1, start_xy[flow2_mask, 1] + 1
                ]
                == flow2_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow2_mask, 0] + 1, start_xy[flow2_mask, 1] + 1
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
    return greedy_expansion(
        start_xy,
        pixel_labels_arr,
        flow_labels,
        flow_labels2,
        flow_xy,
        flow_xy2,
        foreground_mask,
        max_expansion_steps=max_expansion_steps,
    )
