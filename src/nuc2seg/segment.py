import torch
import numpy as np
import geopandas
import pandas
import tqdm
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, softmax

from xenium_utils import pol2cart, create_pixel_geodf, load_nuclei

logger = logging.getLogger(__name__)


def temp_forward(model, x, y, z):
    mask = z > -1
    b = torch.as_tensor(
        np.tile(np.arange(z.shape[0]), (z.shape[1], 1)).T[mask.numpy().astype(bool)]
    )
    W = model.filters(z[mask])
    t_input = torch.Tensor(np.zeros((z.shape[0],) + model.img_shape))
    t_input.index_put_(
        (b, torch.LongTensor(x[mask]), torch.LongTensor(y[mask])), W, accumulate=True
    )
    t_input = torch.Tensor.permute(
        t_input, (0, 3, 1, 2)
    )  # Needs to be Batch x Channels x ImageX x ImageY
    return torch.Tensor.permute(
        model.unet(t_input), (0, 2, 3, 1)
    )  # Map back to Batch x ImageX x Image Y x Classes


def stitch_tile_predictions(model, dataset, tile_buffer=8):
    """TODO: all of the metadata info should be available in dataset."""
    model.eval()

    x_max, y_max = dataset.locations.max(axis=0)
    tile_width, tile_height = dataset[0]["labels"].numpy().shape

    results = np.zeros((x_max + tile_width, y_max + tile_height, dataset.n_classes + 2))
    for idx in tqdm.trange(len(dataset), desc="Stitching tiles"):
        tile = dataset[idx]

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

        # mask_pred = model(x,y,z).detach().numpy().copy()
        mask_pred = (
            temp_forward(model, x[None], y[None], z[None])
            .squeeze(0)
            .detach()
            .numpy()
            .copy()
        )  # TEMP: code is fixed but i am currently using a pretrained model
        foreground_pred = expit(mask_pred[..., 0])
        angles_pred = expit(mask_pred[..., 1]) * 2 * np.pi - np.pi
        class_pred = softmax(mask_pred[..., 2:], axis=-1)

        # Get the location of this tile in the whole slide
        x_start, y_start = location

        # Figure out which parts of the tile to use since the tiles overlap
        x_end_offset, y_end_offset = foreground_pred.shape[:2]
        x_start_offset, y_start_offset = 0, 0
        if x_start > 0:
            x_start_offset += tile_buffer
        if y_start > 0:
            y_start_offset += tile_buffer
        if x_start < x_max:
            x_end_offset -= tile_buffer
        if y_start < y_max:
            y_end_offset -= tile_buffer

        results[
            x_start + x_start_offset : x_start + x_end_offset,
            y_start + y_start_offset : y_start + y_end_offset,
            0,
        ] = foreground_pred[x_start_offset:x_end_offset, y_start_offset:y_end_offset]
        results[
            x_start + x_start_offset : x_start + x_end_offset,
            y_start + y_start_offset : y_start + y_end_offset,
            1,
        ] = angles_pred[x_start_offset:x_end_offset, y_start_offset:y_end_offset]
        results[
            x_start + x_start_offset : x_start + x_end_offset,
            y_start + y_start_offset : y_start + y_end_offset,
            2:,
        ] = class_pred[x_start_offset:x_end_offset, y_start_offset:y_end_offset]

        idx += 1

    return results


def greedy_expansion(
    start_xy,
    pixel_labels_arr,
    flow_labels,
    flow_labels2,
    foreground_mask,
    max_expansion_steps=50,
):
    for step in tqdm.trange(max_expansion_steps, desc="greedy_expansion", unit="step"):
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
            logger.debug("No pixels left to update. Stopping expansion.")
            break

        # Update the filtered pixels to have the assignment of their flow neighbor
        # pixel_labels_arr[start_xy[update_mask,0], start_xy[update_mask,1]] = flow_labels_flat[update_mask]

        # Immediate neighbor flows
        flow1_mask = update_mask & (flow_labels_flat != -1)
        flow1_targets = np.array(flow_labels_flat[flow1_mask])
        flow1_connected = (
            (
                pixel_labels_arr[
                    start_xy[flow1_mask, 0] - 1, start_xy[flow1_mask, 1] - 1
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow1_mask, 0] + 1, start_xy[flow1_mask, 1] - 1
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow1_mask, 0] - 1, start_xy[flow1_mask, 1] + 1
                ]
                == flow1_targets
            )
            | (
                pixel_labels_arr[
                    start_xy[flow1_mask, 0] + 1, start_xy[flow1_mask, 1] + 1
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


def flow_destination(start_xy, angle_preds, magnitude):
    dxdy = np.array(pol2cart(magnitude, angle_preds))[
        :, start_xy[:, 0], start_xy[:, 1]
    ].T
    flow_xy = np.round(start_xy + dxdy).astype(int)
    flow_xy[:, 0] = flow_xy[:, 0].clip(0, x_max).astype(int)
    flow_xy[:, 1] = flow_xy[:, 1].clip(0, y_max).astype(int)
    return flow_xy


def flow_graph_segmentation(
    foreground_preds,
    angle_preds,
    class_preds,
    nuclei_file=None,
    foreground_nucleus_distance=1,
    foreground_threshold=0.5,
    max_expansion_steps=15,
    min_new_cell_area=30,
):
    # Create a dataframe with an entry for every pixel
    img_shape = np.array(foreground_preds.shape)
    x_max, y_max = img_shape - 1
    pixel_labels_arr = np.zeros(img_shape, dtype=int) - 1
    labels_geo_df = create_pixel_geodf(x_max, y_max)

    # Determine where each pixel would flow to next
    start_xy = np.array(
        [labels_geo_df["X"].values, labels_geo_df["Y"].values], dtype=int
    ).T
    flow_xy = flow_destination(start_xy, angle_preds, np.sqrt(2))

    if nuclei_file is not None:
        # Load the nuclei boundaries and assign them unique integer IDs
        nuclei_geo_df = load_nuclei(nuclei_file)

        # Find the nearest nucleus to each pixel
        labels_geo_df = geopandas.sjoin_nearest(
            labels_geo_df, nuclei_geo_df, how="left", distance_col="nucleus_distance"
        )
        labels_geo_df.rename(columns={"index_right": "nucleus_id_xenium"}, inplace=True)

        # Assign pixels roughly on top of nuclei to belong to that nuclei label
        nucleus_mask = labels_geo_df["nucleus_distance"] <= foreground_nucleus_distance


# TODO: fix this to just use connected components
def greedy_cell_segmentation(
    foreground_preds,
    angle_preds,
    class_preds,
    nuclei_file=None,
    foreground_nucleus_distance=1,
    foreground_threshold=0.5,
    max_expansion_steps=15,
    min_new_cell_area=30,
):
    # Create a dataframe with an entry for every pixel
    img_shape = np.array(foreground_preds.shape)
    x_max, y_max = img_shape - 1
    pixel_labels_arr = np.zeros(img_shape, dtype=int) - 1
    labels_geo_df = create_pixel_geodf(x_max, y_max)

    # Determine where each pixel would flow to next
    start_xy = np.array(
        [labels_geo_df["X"].values, labels_geo_df["Y"].values], dtype=int
    ).T
    flow_xy = flow_destination(start_xy, angle_preds, np.sqrt(2))
    flow_labels = np.zeros(img_shape, dtype=int) - 1
    flow_labels[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy[:, 0], flow_xy[:, 1]
    ]

    flow_xy2 = flow_destination(start_xy, angle_preds, np.sqrt(3))
    flow_labels2 = np.zeros(img_shape, dtype=int) - 1
    flow_labels2[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
        flow_xy2[:, 0], flow_xy2[:, 1]
    ]

    # Get the pixels that are sufficiently predicted to be foreground
    foreground_mask = (foreground_preds >= foreground_threshold)[
        start_xy[:, 0], start_xy[:, 1]
    ]

    if nuclei_file is not None:
        # Load the nuclei boundaries and assign them unique integer IDs
        nuclei_geo_df = load_nuclei(nuclei_file)

        # Find the nearest nucleus to each pixel
        labels_geo_df = geopandas.sjoin_nearest(
            labels_geo_df, nuclei_geo_df, how="left", distance_col="nucleus_distance"
        )
        labels_geo_df.rename(columns={"index_right": "nucleus_id_xenium"}, inplace=True)

        # Assign pixels roughly on top of nuclei to belong to that nuclei label
        nucleus_mask = labels_geo_df["nucleus_distance"] <= foreground_nucleus_distance
        pixel_labels_arr[
            labels_geo_df["X"].values[nucleus_mask],
            labels_geo_df["Y"].values[nucleus_mask],
        ] = labels_geo_df["nucleus_label"][nucleus_mask]
        flow_labels[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
            flow_xy[:, 0], flow_xy[:, 1]
        ]
        flow_labels2[start_xy[:, 0], start_xy[:, 1]] = pixel_labels_arr[
            flow_xy2[:, 0], flow_xy2[:, 1]
        ]

        # Greedily expand the cell one pixel at a time
        greedy_expansion(
            start_xy,
            pixel_labels_arr,
            flow_labels,
            flow_labels2,
            foreground_mask,
            max_expansion_steps=max_expansion_steps,
        )

    ##### Segment cells with no nuclei masks
    # Find all the unsegmented foreground regions
    pixel_labels_flat = pixel_labels_arr[start_xy[:, 0], start_xy[:, 1]]
    flow_labels_flat = flow_labels[start_xy[:, 0], start_xy[:, 1]]
    update_mask = foreground_mask & (pixel_labels_flat == -1) & (flow_labels_flat == -1)

    # Connect adjacent pixels where one pixel flows into the next
    rows = (
        start_xy[update_mask, 0] * pixel_labels_arr.shape[1] + start_xy[update_mask, 1]
    )
    cols = flow_xy[update_mask, 0] * pixel_labels_arr.shape[1] + flow_xy[update_mask, 1]
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(update_mask.shape[0], update_mask.shape[0]),
    )

    # Estimate connected subgraphs
    n_connected, connected_labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    # Filter down to segments with enough pixels
    uniques, counts = np.unique(connected_labels[update_mask], return_counts=True)
    uniques = uniques[counts >= min_new_cell_area]
    counts = counts[counts >= min_new_cell_area]

    # Update the pixel labels with the new cells
    n_nuclei = pixel_labels_arr.max() + 1
    for offset, connected_label in enumerate(uniques):
        logger.info(f"Segmenting cells without nuclei ({offset+1}/{len(uniques)}")
        mask = connected_labels == connected_label
        pixel_labels_arr[start_xy[mask, 0], start_xy[mask, 1]] = n_nuclei + offset

    # Post-process to remove island pixels surrounded by the same class on all sides
    island_mask = np.zeros(pixel_labels_arr.shape, dtype=bool)
    island_mask[1:-1, 1:-1] = (
        (pixel_labels_arr[1:-1, 1:-1] == -1)
        & (pixel_labels_arr[:-2, 1:-1] != -1)
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[2:, 1:-1])
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, :-2])
        & (pixel_labels_arr[:-2, 1:-1] == pixel_labels_arr[1:-1, 2:])
    )
    pixel_labels_arr[island_mask] = pixel_labels_arr[:-2, 1:-1][island_mask[1:-1, 1:-1]]

    # Update the flow labels
    flow_labels[start_xy[:, 0], start_xy[:, 1]] = np.maximum(
        pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
        pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
    )
    flow_labels2[start_xy[:, 0], start_xy[:, 1]] = np.maximum(
        pixel_labels_arr[flow_xy[:, 0], flow_xy[:, 1]],
        pixel_labels_arr[flow_xy2[:, 0], flow_xy2[:, 1]],
    )

    # Greedily expand the cell one pixel at a time
    greedy_expansion(
        start_xy,
        pixel_labels_arr,
        flow_labels,
        flow_labels2,
        foreground_mask,
        max_expansion_steps=max_expansion_steps,
    )

    ###### Cell type predictions for each segmented cell
    uniques, inv_map, unique_counts = np.unique(
        pixel_labels_arr[: class_preds.shape[0], : class_preds.shape[1]],
        return_inverse=True,
        return_counts=True,
    )
    class_avgs = np.zeros((len(uniques), class_preds.shape[-1]))
    np.add.at(class_avgs, inv_map, class_preds.reshape(-1, class_preds.shape[-1]))
    class_avgs = class_avgs / unique_counts[:, None]
    class_labels = np.zeros(uniques.max() + 1, dtype=int)
    class_labels[uniques[1:] - 1] = class_avgs[1:].argmax(
        axis=1
    )  # MLE class assignment
    # TODO: should this be an empirical Bayes method? i.e. do we now plug back in the genes and infer a posterior?

    return pixel_labels_arr, class_labels

    segments = pixel_labels_arr

    # Get the nucleus pixel labels
    pixel_nuclei_arr = np.zeros((x_max + 1, y_max + 1), dtype=int) - 1
    nucleus_mask = labels_geo_df["nucleus_distance"] <= foreground_nucleus_distance
    pixel_nuclei_arr[
        labels_geo_df["X"].values[nucleus_mask], labels_geo_df["Y"].values[nucleus_mask]
    ] = labels_geo_df["nucleus_label"][nucleus_mask]

    # Get the distribution of genes in each cell type
    empirical_gene_probs = np.zeros((class_preds.shape[-1], n_genes))
    gene_class_labels = class_labels[
        segments[
            tx_geo_df["x_location"].astype(int), tx_geo_df["y_location"].astype(int)
        ]
        - 1
    ]
    np.add.at(
        empirical_gene_probs,
        (
            gene_class_labels[gene_class_labels >= 0],
            tx_geo_df["gene_id"][gene_class_labels >= 0],
        ),
        1,
    )
    empirical_gene_probs = empirical_gene_probs / empirical_gene_probs.sum(
        axis=1, keepdims=True
    )
    marker_weights = empirical_gene_probs / empirical_gene_probs.mean(
        axis=0, keepdims=True
    )
    markers = tx_geo_df["feature_name"].unique()

    plt.close()
    n_markers = 20
    n_cols = int(np.ceil(np.sqrt(marker_weights.shape[0])))
    n_rows = marker_weights.shape[0] // n_cols
    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    for idx in range(marker_weights.shape[0]):
        i, j = idx // n_cols, idx % n_cols
        axarr[i, j].scatter(
            np.arange(n_markers) + 1,
            marker_weights[idx, np.argsort(marker_weights[idx])[::-1]][:n_markers],
        )
        axarr[i, j].set_xticks(np.arange(n_markers) + 1)
        axarr[i, j].set_xticklabels(
            markers[np.argsort(marker_weights[idx])[::-1]][:n_markers],
            rotation="vertical",
        )
    plt.tight_layout()
    plt.savefig("plots/markers.pdf", bbox_inches="tight")
    plt.close()

    plt.close()
    x_regions = np.array([(2000, 2100), (350, 450), (80, 180), (3300, 3400)])
    y_regions = np.array([(2000, 2100), (3350, 3450), (3270, 3370), (4400, 4500)])
    fig, axarr = plt.subplots(4, len(regions), sharex=True, sharey=True)
    for i, ((x_start, x_stop), (y_start, y_stop)) in enumerate(
        zip(x_regions, y_regions)
    ):
        local_nuclei = np.array(
            pixel_nuclei_arr[x_start:x_stop, y_start:y_stop], dtype=float
        )
        local_foreground = foreground_preds[x_start:x_stop, y_start:y_stop]
        local_angles = angle_preds[x_start:x_stop, y_start:y_stop]
        local_segments = np.array(segments[x_start:x_stop, y_start:y_stop], dtype=float)
        local_celltypes = np.full(local_segments.shape, np.nan) - 1
        local_celltypes[local_segments >= 0] = class_labels[
            local_segments[local_segments >= 0].astype(int)
        ]
        axarr[1, i].imshow(
            local_foreground, vmin=0, vmax=1, cmap="coolwarm", interpolation="none"
        )

        for xi in range(local_foreground.shape[0]):
            for yi in range(local_foreground.shape[1]):
                if local_foreground[xi, yi] >= 0.5:
                    dx, dy = pol2cart(0.5, local_angles[xi, yi])
                    axarr[1, i].arrow(yi + 0.5, xi + 0.5, dy, dx, width=0.07, alpha=0.5)

        local_nuclei[local_nuclei >= 0] = local_nuclei[local_nuclei >= 0] + len(
            np.unique(local_segments)
        )
        local_segments[local_segments >= 0] = local_segments[local_segments >= 0] + len(
            np.unique(local_segments)
        )
        local_labels = np.unique(local_segments)
        if len(local_labels) > 20:
            local_labels = local_labels[local_labels >= 0]
            for j, c in enumerate(
                np.random.choice(local_labels, size=len(local_labels), replace=False)
            ):
                local_segments[local_segments == c] = j
                local_nuclei[local_nuclei == c] = j
        local_segments[local_segments < 0] = np.nan
        local_nuclei[local_nuclei < 0] = np.nan
        axarr[2, i].imshow(local_segments, cmap="tab20b", interpolation="none")
        axarr[0, i].imshow(local_nuclei, cmap="tab20b", interpolation="none")
        axarr[3, i].imshow(local_celltypes, cmap="tab20b", interpolation="none")

        axarr[0, i].set_title("Nuclei segmentation")
        axarr[1, i].set_title("Foreground predictions")
        axarr[2, i].set_title("Angle-based segmentation")
        axarr[3, i].set_title("Cell type labels")
    plt.tight_layout()
    plt.show()


def save_cell_matrix(pixel_labels_arr, tx_geo_df, outfile):
    n_cells = pixel_labels_arr.max()
    n_genes = tx_geo_df["gene_id"].max() + 1
    counts = np.zeros((n_cells, n_genes), dtype=int)

    # Get the coordinates and gene ID
    X, Y, gene_id = (
        tx_geo_df["x_location"].values.astype(int),
        tx_geo_df["y_location"].astype(int),
        tx_geo_df["gene_id"],
    )

    # Filter to only transcripts inside a cell
    assigned = pixel_labels_arr[X, Y] > 0
    X, Y, gene_id = X[assigned], Y[assigned], gene_id[assigned]

    # Accumulate transcripts
    np.add.at(counts, (pixel_labels_arr[X, Y] - 1, gene_id), 1)

    # Determine the centroid location of each cell
    x_centers, y_centers, n_pixels = (
        np.zeros(n_cells),
        np.zeros(n_cells),
        np.zeros(n_cells),
    )
    grid_xy = np.array(
        np.meshgrid(
            np.arange(pixel_labels_arr.shape[0]), np.arange(pixel_labels_arr.shape[1])
        )
    ).T.reshape(-1, 2)
    assigned = pixel_labels_arr.reshape(-1) > 0
    np.add.at(
        x_centers, pixel_labels_arr.reshape(-1)[assigned] - 1, grid_xy[assigned, 0]
    )
    np.add.at(
        y_centers, pixel_labels_arr.reshape(-1)[assigned] - 1, grid_xy[assigned, 1]
    )
    np.add.at(n_pixels, pixel_labels_arr.reshape(-1)[assigned] - 1, 1)
    centroids = np.array(
        [
            x_centers / np.clip(n_pixels, 1, None),
            y_centers / np.clip(n_pixels, 1, None),
            n_pixels,
        ]
    ).T

    # Build a dataframe
    df = {
        "centroid_x": x_centers / np.clip(n_pixels, 1, None),
        "centroid_y": y_centers / np.clip(n_pixels, 1, None),
        "cell_area": n_pixels,
    }
    for idx, gene_name in enumerate(tx_geo_df["feature_name"].unique()):
        df[gene_name] = counts[:, idx]

    # Save to file
    pandas.DataFrame(df).to_csv(outfile, index=False)
