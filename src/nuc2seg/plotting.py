import json
import os.path
from typing import Optional

import geopandas
import numpy as np
import pandas
import seaborn as sns
import shapely
import tqdm
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import (
    GeoJSONDataSource,
    ColumnDataSource,
    CheckboxGroup,
    CustomJS,
    WheelZoomTool,
    PanTool,
    ResetTool,
    SaveTool,
    TapTool,
    InlineStyleSheet,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.resources import INLINE
from matplotlib import cm, gridspec, animation
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from shapely import box

from nuc2seg.data import (
    Nuc2SegDataset,
    ModelPredictions,
    CelltypingResults,
)
from nuc2seg.preprocessing import pol2cart
from nuc2seg.segment import greedy_cell_segmentation
from nuc2seg.utils import (
    transform_shapefile_to_rasterized_space,
    bbox_geometry_to_rasterized_slice,
)


def plot_tiling(bboxes, output_path):
    polygons = [box(*x) for x in bboxes]
    gdf = geopandas.GeoDataFrame(geometry=polygons)

    fig, ax = plt.subplots()

    gdf.boundary.plot(ax=ax, color="red", alpha=0.5)

    fig.savefig(output_path)
    plt.close(fig)


def plot_labels(ax, dataset: Nuc2SegDataset, bbox=None):
    label_plot = dataset.labels.copy()
    transcripts = dataset.transcripts.copy()

    if bbox is not None:
        label_plot = label_plot[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask_x = (transcripts[:, 0] >= bbox[0]) & (transcripts[:, 0] < bbox[2])
        mask_y = (transcripts[:, 1] >= bbox[1]) & (transcripts[:, 1] < bbox[3])
        mask = mask_x & mask_y
        transcripts = transcripts[mask]

        transcripts[:, 0] = transcripts[:, 0] - bbox[0]
        transcripts[:, 1] = transcripts[:, 1] - bbox[1]

    # Created fixed colormap for labels where 0 is background, 1 is border and 2 is nucleus
    nucleus_color = cm.copper(200)[:3]  # Color for 'Nucleus'
    border_color = cm.copper(128)[:3]  # Color for 'Border'
    background_color = cm.copper(0)[:3]  # Color for 'Background'

    ax.set_title("Labels and transcripts")

    label_plot_image = np.zeros((label_plot.shape[0], label_plot.shape[1], 3))

    label_plot_image[label_plot == 0] = background_color
    label_plot_image[label_plot == -1] = border_color
    label_plot_image[label_plot > 0] = nucleus_color

    ax.imshow(label_plot_image.transpose(1, 0, 2))

    # Add legend with square patches of color to the right of the plot area
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=nucleus_color,
                markersize=10,
                label="Nucleus",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=border_color,
                markersize=10,
                label="Unlabeled",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=background_color,
                markersize=10,
                label="Background",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    ax.scatter(
        x=transcripts[:, 0],
        y=transcripts[:, 1],
        color="red",
        zorder=100,
        s=1,
        alpha=0.4,
    )


def plot_angles_quiver(
    ax,
    dataset: Nuc2SegDataset,
    predictions: ModelPredictions,
    bbox=None,
):
    angles = predictions.angles
    labels = dataset.labels

    if bbox is not None:
        angles = angles[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        labels = labels[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        nuclei = labels > 0
        mask = labels == -1
    else:
        nuclei = labels > 0
        mask = labels == -1

    nuclei = nuclei.T

    imshow_data = np.zeros((nuclei.shape[0], nuclei.shape[1], 4)).astype(float)

    for i in range(mask.T.shape[0]):
        for j in range(mask.T.shape[1]):
            if not mask.T[i, j]:
                imshow_data[i, j, :] = np.array([0.45, 0.57, 0.70, 0.5]).astype(float)

    ax.imshow(imshow_data)
    norm = mcolors.Normalize(vmin=angles.min(), vmax=angles.max())

    for xi in range(nuclei.shape[1]):
        for yi in range(nuclei.shape[0]):
            if mask[xi, yi]:
                dx, dy = pol2cart(0.5, angles[xi, yi])
                ax.arrow(
                    xi,
                    yi,
                    dx,
                    dy,
                    color=cm.hsv(norm(angles[xi, yi])),
                    width=(1 / nuclei.shape[1] * 5),
                )
    ax.set_title("Predicted angles and segmentation")

    legend_handles = []
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="Nucleus",
        )
    )

    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=(0.45, 0.57, 0.70, 0.5),
            markersize=10,
            label="Background",
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )


def plot_foreground(ax, predictions: ModelPredictions, bbox=None):
    foreground = predictions.foreground

    if bbox is not None:
        foreground = foreground[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    ax.set_title("Predicted foreground")
    return ax.imshow(
        foreground.T, vmin=0, vmax=1, cmap="coolwarm", interpolation="none"
    )


def update_projection(ax_dict, ax_key, projection="3d", fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = ax_dict[ax_key].get_subplotspec().get_geometry()
    ax_dict[ax_key].remove()
    ax_dict[ax_key] = fig.add_subplot(rows, cols, start + 1, projection=projection)


def plot_preprocessing(dataset: Nuc2SegDataset, output_path: str):
    layout = """
    A
    B
    C
    """
    fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(10, 30))
    plot_labels(ax["A"], dataset, bbox=None)

    plot_angles_quiver(
        ax=ax["B"],
        predictions=dataset.angles,
        bbox=None,
        dataset=dataset,
    )

    plot_nuclei_labels(ax["C"], dataset)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_nuclei_labels(ax, dataset: Nuc2SegDataset):
    imshow_data = np.zeros(
        (dataset.classes.T.shape[0], dataset.classes.T.shape[1], 4)
    ).astype(float)

    for i in range(dataset.classes.T.shape[0]):
        for j in range(dataset.classes.T.shape[1]):
            if dataset.labels.T[i, j] > 0:
                # get color from cm.tab10
                color = cm.tab10(dataset.classes.T[i, j] - 1)[:3]
                imshow_data[i, j, :] = np.array(
                    [color[0], color[1], color[2], 1.0]
                ).astype(float)

    ax.set_title("Labels")
    ax.imshow(imshow_data, interpolation="none")

    legend_handles = []
    for i in range(np.unique(dataset.classes)):
        if i == 0:
            continue
        i = i - 1
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cm.tab10(i)[:3],
                markersize=10,
                label=f"Class {i}",
            )
        )
    # put legend upper right
    ax.legend(
        handles=legend_handles,
        loc="upper right",
    )


def plot_model_predictions(
    dataset: Nuc2SegDataset,
    prior_segmentation_gdf: geopandas.GeoDataFrame,
    segmentation_gdf: geopandas.GeoDataFrame,
    model_predictions: ModelPredictions,
    output_path: str,
    bbox: shapely.Polygon,
):
    layout = """
    A
    B
    C
    """

    x1, y1, x2, y2 = bbox_geometry_to_rasterized_slice(
        bbox=bbox, resolution=dataset.resolution, sample_area=dataset.bbox
    )

    prior_segmentation_transformed = transform_shapefile_to_rasterized_space(
        gdf=prior_segmentation_gdf,
        sample_area=bbox.bounds,
        resolution=dataset.resolution,
    )

    model_predictions = model_predictions.clip((x1, y1, x2, y2))
    dataset = dataset.clip((x1, y1, x2, y2))

    fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(10, 30))
    plot_angles_quiver(
        ax=ax["B"],
        predictions=model_predictions,
        bbox=None,
        dataset=dataset,
    )

    plot_monocolored_seg_outlines(
        ax=ax["B"],
        gdf=prior_segmentation_transformed,
        color="black",
    )

    plot_labels(ax["A"], dataset, bbox=None)
    im = plot_foreground(ax["C"], model_predictions, bbox=None)
    fig.colorbar(im, ax=ax["C"])
    plot_monocolored_seg_outlines(ax=ax["C"], gdf=prior_segmentation_transformed)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_class_predictions(
    dataset: Nuc2SegDataset,
    prior_segmentation_gdf: geopandas.GeoDataFrame,
    segmentation_gdf: geopandas.GeoDataFrame,
    model_predictions: ModelPredictions,
    output_path: str,
    bbox: shapely.Polygon,
):
    fig, ax = plt.subplots(
        nrows=dataset.n_classes + 1,
        ncols=1,
        figsize=(10, 10 * (dataset.n_classes + 1)),
        dpi=100,
    )

    x1, y1, x2, y2 = bbox_geometry_to_rasterized_slice(
        bbox=bbox, resolution=dataset.resolution, sample_area=dataset.bbox
    )

    prior_segmentation_transformed = transform_shapefile_to_rasterized_space(
        gdf=prior_segmentation_gdf,
        sample_area=bbox.bounds,
        resolution=dataset.resolution,
    )
    segmentation_transformed = transform_shapefile_to_rasterized_space(
        gdf=segmentation_gdf,
        sample_area=bbox.bounds,
        resolution=dataset.resolution,
    )

    model_predictions = model_predictions.clip((x1, y1, x2, y2))
    dataset = dataset.clip((x1, y1, x2, y2))

    for i in range(dataset.n_classes):
        ax[i + 1].set_title(f"Class {i}")
        im = ax[i + 1].imshow(
            model_predictions.classes[:, :, i].T,
            cmap="coolwarm",
            vmin=model_predictions.classes.min(),
            vmax=model_predictions.classes.max(),
            interpolation="none",
        )

        fig.colorbar(im, ax=ax[i + 1])
        plot_monocolored_seg_outlines(
            ax=ax[i + 1],
            gdf=prior_segmentation_transformed,
        )

        plot_monocolored_seg_outlines(
            ax=ax[i + 1],
            gdf=segmentation_transformed,
        )

        plot_monocolored_seg_outlines(
            ax=ax[i + 1],
            gdf=segmentation_transformed[
                (segmentation_transformed["celltype_assignment"] == i)
                & (
                    segmentation_transformed["celltype_assignment"]
                    == segmentation_transformed["unet_celltype_assignment"]
                )
            ],
            color="yellow",
        )

        plot_monocolored_seg_outlines(
            ax=ax[i + 1],
            gdf=segmentation_transformed[
                (segmentation_transformed["celltype_assignment"] == i)
                & (
                    segmentation_transformed["celltype_assignment"]
                    != segmentation_transformed["unet_celltype_assignment"]
                )
            ],
            color="red",
        )

        # add class legend
        legend_handles = []
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="yellow",
                markerfacecolor="yellow",
                markersize=10,
                label=f"True Pos",
            )
        )

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="red",
                markerfacecolor="red",
                markersize=10,
                label=f"False Pos",
            )
        )
        # put legend upper right
        ax[i + 1].legend(
            handles=legend_handles,
            loc="upper right",
        )

    imshow_data = np.zeros(
        (dataset.classes.T.shape[0], dataset.classes.T.shape[1], 4)
    ).astype(float)

    for i in range(dataset.classes.T.shape[0]):
        for j in range(dataset.classes.T.shape[1]):
            if dataset.labels.T[i, j] > 0:
                # get color from cm.tab10
                color = cm.tab10(dataset.classes.T[i, j] - 1)[:3]
                imshow_data[i, j, :] = np.array(
                    [color[0], color[1], color[2], 1.0]
                ).astype(float)

    ax[0].set_title("Labels")
    ax[0].imshow(imshow_data, interpolation="none")

    # add class legend
    legend_handles = []
    for i in range(dataset.n_classes):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cm.tab10(i)[:3],
                markersize=10,
                label=f"Class {i}",
            )
        )
    # put legend upper right
    ax[0].legend(
        handles=legend_handles,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_final_segmentation(nuclei_gdf, segmentation_gdf, output_path):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    ax.invert_yaxis()
    segmentation_gdf.plot(ax=ax, color="blue", edgecolor="lightgray", linewidth=0.1)
    nuclei_gdf.plot(ax=ax, color="red")

    fig.savefig(output_path)
    plt.close(fig)


def plot_multicolored_seg_outlines(
    ax,
    gdf: geopandas.GeoDataFrame,
):
    edge_colors = [cm.tab10(i % 10) for i in range(len(gdf))]
    gdf.plot(ax=ax, facecolor=(0, 0, 0, 0), edgecolor=edge_colors, linewidth=0.5)


def plot_monocolored_seg_outlines(ax, gdf: geopandas.GeoDataFrame, color="black"):
    gdf.plot(ax=ax, facecolor=(0, 0, 0, 0), edgecolor=color, linewidth=1.0)


def plot_segmentation_comparison(
    seg_a,
    seg_b,
    nuclei,
    output_path,
    transcripts=None,
    seg_a_name="sega",
    seg_b_name="segb",
    bbox=None,
):
    if bbox:
        seg_a = seg_a[seg_a.geometry.within(bbox)]
        seg_b = seg_b[seg_b.geometry.within(bbox)]
        nuclei = nuclei[nuclei.geometry.within(bbox)]
        if transcripts is not None:
            transcripts = transcripts[transcripts.geometry.within(bbox)].copy()
            transcripts["geometry"] = transcripts.geometry.buffer(0.05)
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    ax.invert_yaxis()
    seg_a.plot(ax=ax, color="blue", alpha=0.5, edgecolor="white", linewidth=0.5)
    seg_b.plot(ax=ax, color="red", alpha=0.5, edgecolor="white", linewidth=0.5)
    nuclei.plot(ax=ax, color="black", alpha=0.5, edgecolor="white", linewidth=0.5)
    if transcripts is not None:
        transcripts.plot(ax=ax, color="green", alpha=0.8, linewidth=0)

    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label=seg_a_name,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label=seg_b_name,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=10,
                label="Nucleus",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1, 0.5),
    )

    fig.savefig(output_path)
    plt.close(fig)


def plot_segmentation_class_assignment(
    segmentation_gdf, output_path, cat_column="class_assignment"
):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    ax.invert_yaxis()
    segmentation_gdf.plot(
        ax=ax,
        categorical=True,
        column=cat_column,
        legend=True,
        cmap="tab20",
        edgecolor="lightgray",
        linewidth=0.1,
    )
    fig.savefig(output_path)
    plt.close(fig)


def plot_gene_choropleth(segmentation_gdf, adata, gene_name, output_path, log=True):
    segmentation_gdf = segmentation_gdf.reset_index(names="_index")
    df = adata[:, gene_name].to_df()

    df.reset_index(inplace=True, names="_index")
    df["_index"] = df["_index"].astype(int)

    joined_gdf = pandas.merge(
        segmentation_gdf, df, left_on="_index", right_on="_index", how="left"
    )

    if (gene_name + "_y") in joined_gdf:
        joined_gdf[gene_name] = joined_gdf[gene_name + "_y"]

    if log:
        joined_gdf[gene_name] = np.log(joined_gdf[gene_name] + 1)

    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    ax.invert_yaxis()
    joined_gdf.plot(
        ax=ax,
        categorical=False,
        column=gene_name,
        legend=True,
        cmap="coolwarm",
        edgecolor="lightgray",
        linewidth=0.1,
    )
    fig.savefig(output_path)
    plt.close(fig)


def celltype_histogram(segmentation_gdf, output_path, cat_column="celltype_assignment"):
    layout = "AB"

    fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(20, 8))

    sns.histplot(
        ax=ax["A"],
        data=segmentation_gdf,
        x=cat_column,
        hue=cat_column,
        palette="tab20",
        legend=False,
    )
    ax["A"].set_title("Number of Cells per Celltype")
    ax["A"].set_xticks(range(segmentation_gdf[cat_column].nunique()))
    ax["A"].set_ylabel("# Cells")

    area_df = segmentation_gdf.groupby(cat_column)["area"].sum().reset_index()

    sns.barplot(
        ax=ax["B"], data=area_df, x=cat_column, y="area", palette="tab20", legend=False
    )
    ax["B"].set_title("Total Segmented Area per Celltype")
    ax["B"].set_xticks(range(segmentation_gdf[cat_column].nunique()))
    ax["B"].set_ylabel("Area (μm^2)")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def celltype_area_violin(
    segmentation_gdf, output_path, cat_column="celltype_assignment"
):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title("Distribution of Segmented Area per Celltype")
    sns.violinplot(
        ax=ax,
        data=segmentation_gdf,
        x=cat_column,
        y="area",
        legend=False,
        inner="quart",
        palette="tab20",
    )

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_celltype_estimation_results(
    aic_scores,
    bic_scores,
    final_expression_profiles,
    final_prior_probs,
    relative_expression,
    n_components,
    output_dir,
):
    # create output_dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a double line plot with aic and bic scores
    fig, ax = plt.subplots(figsize=(10, 10))

    error = np.stack(
        [
            np.abs(aic_scores.mean(axis=0) - aic_scores.min(axis=0)),
            np.abs(aic_scores.mean(axis=0) - aic_scores.max(axis=0)),
        ]
    )

    # add min/max error bars
    ax.errorbar(
        n_components,
        aic_scores.mean(axis=0),
        yerr=error,
        label="AIC",
        alpha=0.5,
    )

    error = np.stack(
        [
            bic_scores.mean(axis=0) - bic_scores.min(axis=0),
            bic_scores.max(axis=0) - bic_scores.mean(axis=0),
        ]
    )

    ax.errorbar(
        n_components,
        bic_scores.mean(axis=0),
        yerr=error,
        label="BIC",
        alpha=0.5,
    )

    arg_min = np.argmin(bic_scores.min(axis=0))
    min_value = bic_scores.min(axis=0)[arg_min]
    best_n_celltypes = n_components[arg_min]

    # add star to plot at best n_celltypes
    ax.annotate(
        text=f"Best N Celltypes:\n{best_n_celltypes}",
        xy=(best_n_celltypes, min_value),
    )
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Score")
    ax.set_title("AIC and BIC score ranges over multiple trials")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "aic_bic_scores.pdf"))
    plt.close(fig)

    # Create bar plot with the expression profiles
    for idx, expression_profile in enumerate(final_expression_profiles):

        n_celltypes = expression_profile.shape[0]

        fig, ax = plt.subplots(
            nrows=n_celltypes, figsize=(10, 10), sharex=True, sharey=True
        )

        for i in range(n_celltypes):
            ax[i].bar(range(len(expression_profile[i])), expression_profile[i])

        ax[-1].set_xlabel("Gene")
        ax[0].set_ylabel("Expression")
        fig.suptitle(f"Expression profiles for k={n_celltypes}")
        fig.savefig(
            os.path.join(output_dir, f"expression_profiles_k={n_celltypes}.pdf")
        )
        fig.tight_layout()
        plt.close(fig)

        fig, ax = plt.subplots(
            nrows=n_celltypes, figsize=(10, 10), sharex=True, sharey=True
        )

        for i in range(n_celltypes):
            ax[i].bar(
                range(len(relative_expression[idx][i])), relative_expression[idx][i]
            )

        ax[-1].set_xlabel("Gene")
        ax[0].set_ylabel("Relative expression")
        fig.suptitle(f"Relative expression profiles for k={n_celltypes}")
        fig.savefig(
            os.path.join(
                output_dir, f"relative_expression_profiles_k={n_celltypes}.pdf"
            )
        )
        fig.tight_layout()
        plt.close(fig)

        # Create bar plot with the prior probabilities
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(range(len(final_prior_probs[idx])), final_prior_probs[idx])
        ax.set_xlabel("Cell type")
        ax.set_ylabel("Prior probability")
        ax.set_title(f"Prior probabilities for k={n_celltypes}")
        fig.savefig(os.path.join(output_dir, f"prior_probs_k={n_celltypes}.pdf"))
        fig.tight_layout()
        plt.close(fig)


def plot_foreground_background_benchmark(
    foreground_intensities, background_intensities, output_path
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(foreground_intensities, bins=100, alpha=0.5, label="Foreground")
    ax.hist(background_intensities, bins=100, alpha=0.5, label="Background")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Foreground and background IF intensity distributions")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def plot_segmentation_avg_intensity_distribution(
    nuc2seg_intensities, other_intensities, output_path, other_method_name="Xenium"
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(nuc2seg_intensities, bins=100, alpha=0.5, label="Nuc2Seg")
    ax.hist(other_intensities, bins=100, alpha=0.5, label=other_method_name)
    ax.set_xlabel("Avg Intensity of Cell Segment")
    ax.set_ylabel("Frequency")
    ax.set_title("Average segment intensity distributions")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def plot_segmentation_size_distribution(
    nuc2seg_intensities, other_intensities, output_path, other_method_name="Xenium"
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(nuc2seg_intensities, bins=100, alpha=0.5, label="Nuc2Seg")
    ax.hist(other_intensities, bins=100, alpha=0.5, label=other_method_name)
    ax.set_xlabel("Segment # of Pixels")
    ax.set_ylabel("Frequency")
    ax.set_title("Average segment size distributions")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def foreground_background_boxplot(
    nuc2seg_foreground_intensities,
    nuc2seg_background_intensities,
    other_method_foreground_intensities,
    other_method_background_intensities,
    output_path,
    other_method_name="Xenium",
):
    import seaborn as sns

    sns.set_theme(style="ticks", palette="pastel")

    df = pandas.DataFrame()
    df["intensity"] = np.concatenate(
        [
            nuc2seg_foreground_intensities,
            nuc2seg_background_intensities,
            other_method_foreground_intensities,
            other_method_background_intensities,
        ]
    )

    df["method"] = (
        ["Nuc2Seg"] * len(nuc2seg_foreground_intensities)
        + ["Nuc2Seg"] * len(nuc2seg_background_intensities)
        + [other_method_name] * len(other_method_foreground_intensities)
        + [other_method_name] * len(other_method_background_intensities)
    )
    df["class"] = (
        ["Foreground"] * len(nuc2seg_foreground_intensities)
        + ["Background"] * len(nuc2seg_background_intensities)
        + ["Foreground"] * len(other_method_foreground_intensities)
        + ["Background"] * len(other_method_background_intensities)
    )
    # Draw a nested boxplot to show bills by day and time

    sns.boxplot(
        x="method",
        y="intensity",
        hue="class",
        data=df,
        fill=False,
        gap=0.1,
        showfliers=False,
    )
    plt.savefig(output_path)


def rank_genes_groups_plot(
    celltyping_results: CelltypingResults,
    k: int,
    output_path: str,
    n_genes: int = 15,
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    ax=None,
):
    k_idx = celltyping_results.n_component_values.tolist().index(k)
    n_panels_per_row = ncols
    if n_genes < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received n_genes={n_genes}."
        )

    group_names = [f"{i}" for i in range(k)]
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(
            n_panels_x * 3,
            n_panels_y * 3,
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.inf
    ymax = -np.inf
    for celltype_idx in range(k):
        gene_names = celltyping_results.gene_names
        scores = celltyping_results.relative_expression[k_idx][celltype_idx, :]
        sorted_order = np.argsort(scores)[::-1]

        scores = scores[sorted_order][:n_genes]
        gene_names = gene_names[sorted_order][:n_genes]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            if ax0 is None:
                ax = fig.add_subplot(gs[celltype_idx])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[celltype_idx], sharey=ax0)
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[celltype_idx])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_genes - 0.1)

        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax.text(
                ig,
                scores[ig],
                gene_name,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=fontsize,
            )

        ax.set_title(f"{celltype_idx}")
        if celltype_idx >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel("ranking")

        # print the 'score' label only on the first panel per row.
        if celltype_idx % n_panels_x == 0:
            ax.set_ylabel("score")

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, pad_inches=0.5)


def plot_class_probabilities_image(
    model_predictions: ModelPredictions,
    segmentation_shapes: geopandas.GeoDataFrame,
    nuclei_shapes: geopandas.GeoDataFrame,
    output_dir: str,
):
    mask = ((model_predictions.foreground > 0.5).T).astype(int)
    for celltype_idx in tqdm.trange(model_predictions.classes.shape[2]):
        data = model_predictions.classes[..., celltype_idx].copy().T * mask

        fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)

        ax.imshow(data, cmap="Blues", interpolation="none")
        ax.invert_yaxis()
        ax.set_title(f"Pixel Probabilities for Celltype {celltype_idx}")

        segmentation_shapes.plot(
            ax=ax,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor=(199.0 / 255.0, 161 / 255.0, 155 / 255.0, 1.0),
            linewidth=0.1,
        )
        nuclei_shapes.plot(
            ax=ax, facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor="black", linewidth=0.1
        )

        fig.savefig(
            os.path.join(output_dir, f"celltype_probabilities_{celltype_idx}.png")
        )
        plt.close(fig)


class SegmentationPlotter:
    def __init__(self):
        self.frames = []

    def __call__(self, step_idx: int, labels: np.array):
        self.frames.append((step_idx, labels))

    def generate_imshow_array(self, labels, initial_labels):
        imshow_arr = np.ones((labels.shape[0], labels.shape[1], 3)).astype(float)
        imshow_arr[labels > 0] = np.array([0, 0, 1.0])
        imshow_arr[initial_labels > 0] = np.array([1.0, 0, 0])
        return imshow_arr

    def plot(self, output_path):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        image = None
        initial_labels = None

        frames = []

        for step_idx, labels in self.frames:
            if step_idx == 0:
                initial_labels = labels
            imshow_arr = self.generate_imshow_array(labels, initial_labels)
            if step_idx == 0:
                image = ax.imshow(imshow_arr)
            frames.append(imshow_arr)

        def update(frame):
            image.set_array(frame)
            return [image]

        ani = animation.FuncAnimation(
            fig, update, frames=frames, blit=True, interval=200
        )

        ani.save(output_path, writer="ffmpeg", fps=5)
        plt.close(fig)


def plot_greedy_cell_segmentation(
    dataset: Nuc2SegDataset,
    predictions: ModelPredictions,
    prior_probs: np.array,
    expression_profiles: np.array,
    segment_id: int,
    output_path: str,
    window_size=30,
):
    """
    Plot the greedy cell segmentation for a specific segment_id
    """
    mask = dataset.labels == segment_id
    background = (predictions.foreground > 0.5).astype(int)
    # get bounding box of true values
    x, y = np.where(mask)
    bbox = (
        max(x.min() - window_size, 0),
        max(y.min() - window_size, 0),
        min(x.max() + window_size, dataset.labels.shape[0]),
        min(y.max() + window_size, dataset.labels.shape[1]),
    )

    dataset_clipped = dataset.clip(bbox)
    predictions_clipped = predictions.clip(bbox)

    plotting_callback = SegmentationPlotter()

    greedy_cell_segmentation(
        dataset=dataset_clipped,
        predictions=predictions_clipped,
        prior_probs=prior_probs,
        expression_profiles=expression_profiles,
        use_labels=True,
        plotting_callback=plotting_callback,
    )

    plotting_callback.plot(output_path)


def create_interactive_segmentation_comparison(
    polygon_gdfs,
    names,
    points_gdf,
    point_layer_name="Show Tx",
    title="Segmentation Viewer",
    output_path="geoplot.html",
):
    """
    Create an interactive Bokeh plot to display multiple polygon GeoDataFrames with radio button selection
    and optional point data with checkbox control.

    Parameters:
    -----------
    polygon_gdfs : list of GeoDataFrames
        List of GeoDataFrames containing polygon data to visualize
    names : list of str
        Names for each polygon GeoDataFrame to display in radio buttons
    points_gdf : GeoDataFrame, optional
        GeoDataFrame containing point data to overlay
    point_layer_name : str
        Name for the point layer checkbox
    title : str
        Title for the plot
    output_path : str
        Path where the HTML file will be saved
    """
    # Convert Polygon GeoDataFrames to GeoJSONDataSource format
    geojson_sources = []
    for gdf in polygon_gdfs:
        geojson = json.loads(gdf.to_json())
        geojson_source = GeoJSONDataSource(geojson=json.dumps(geojson))
        geojson_sources.append(geojson_source)

    # Convert Points GeoDataFrame to GeoJSONDataSource if provided
    points_df = pandas.DataFrame(
        {"x": points_gdf.geometry.x, "y": points_gdf.geometry.y}
    )

    points_source = ColumnDataSource(points_df)

    # Create figure with explicit tools
    tools = [PanTool(), WheelZoomTool(), ResetTool(), SaveTool(), TapTool()]

    p = figure(title=title, height=1000, width=1000, tools=tools)
    p.title.text_font_size = "48pt"
    p.axis.visible = False
    p.grid.visible = False

    # Create polygon renderers
    polygon_renderers = []
    colors = Category10[10][: len(polygon_gdfs)]
    colors_with_alpha = [
        f"rgba({int(int(color[1:3], 16))}, {int(int(color[3:5], 16))}, {int(int(color[5:7], 16))}, 0.7)"
        for color in colors
    ]

    for source, color in zip(geojson_sources, colors):
        renderer = p.patches(
            "xs",
            "ys",
            fill_color=color,
            fill_alpha=0.7,
            line_color="black",
            line_width=0.5,
            source=source,
        )
        polygon_renderers.append(renderer)
        renderer.visible = False

        # Add tap callback to hide clicked shapes
        tap_callback = CustomJS(
            args=dict(source=source),
            code="""
                        // Get the clicked feature index
                        const ind = cb_obj.indices[0];
                        if (ind !== undefined) {
                            // Get the current visible data
                            const visible = source.data['visible'];

                            // Toggle visibility
                            visible[ind] = 0;  // Hide the clicked shape

                            // Trigger a data change
                            source.change.emit();
                        }
                    """,
        )
        renderer.data_source.selected.js_on_change("indices", tap_callback)

    # Make first polygon renderer visible by default
    polygon_renderers[0].visible = True

    # Create point renderer if points provided
    print(points_source)
    point_renderer = p.scatter(
        "x", "y", source=points_source, size=1.5, color="black", alpha=0.5
    )
    point_renderer.visible = False

    # Create radio buttons for polygons
    checkbox_styles = []
    for i, (name, color) in enumerate(zip(names, colors_with_alpha)):
        checkbox_styles.append(
            f"""
            .bk-input-group label:nth-child({i + 1}) {{
                font-size: 30px;
                font-weight: bold;
                padding: 5px 10px;
                background-color: {color};
                border-radius: 4px;
                margin: 5px 0;
                display: inline-block;
            }}
            """
        )

    # Create polygon checkbox with styles
    polygon_stylesheet = InlineStyleSheet(css="\n".join(checkbox_styles))
    polygon_checkbox = CheckboxGroup(
        labels=names, active=[], stylesheets=[polygon_stylesheet]
    )

    tx_stylesheet = InlineStyleSheet(
        css="""
        label {
            font-size: 30px;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 5px 0;
            display: inline-block;
            border: 1px solid black;
        }
    """
    )
    # Create checkbox for points if provided
    checkbox = CheckboxGroup(
        labels=[point_layer_name], active=[], stylesheets=[tx_stylesheet]
    )

    # Create JavaScript callback for radio buttons
    polygon_callback = CustomJS(
        args={"renderers": polygon_renderers},
        code="""
            // Update visibility based on checked boxes
            const active = cb_obj.active;
            renderers.forEach((r, i) => {
                r.visible = active.includes(i);
            });
            """,
    )

    polygon_checkbox.js_on_change("active", polygon_callback)

    # Create JavaScript callback for checkbox if points provided
    if checkbox and point_renderer:
        checkbox_callback = CustomJS(
            args={"renderer": point_renderer},
            code="""
            // Toggle point layer visibility
            renderer.visible = cb_obj.active.includes(0);
            """,
        )
        checkbox.js_on_change("active", checkbox_callback)

    # Create layout

    layout = row(p, column(polygon_checkbox, checkbox))
    # Configure output to save as standalone HTML
    output_file(output_path, title=title, mode="inline")

    # Save the plot as HTML with embedded data
    save(layout, filename=output_path, title=title, resources=INLINE)

    return layout
