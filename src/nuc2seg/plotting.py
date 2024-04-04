import os.path

import geopandas
import numpy as np
import pandas
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from shapely import box
import seaborn as sns
from nuc2seg.data import (
    Nuc2SegDataset,
    ModelPredictions,
    SegmentationResults,
    CelltypingResults,
)
from nuc2seg.preprocessing import pol2cart


def plot_tiling(bboxes, output_path):
    polygons = [box(*x) for x in bboxes]
    gdf = geopandas.GeoDataFrame(geometry=polygons)

    fig, ax = plt.subplots()

    gdf.boundary.plot(ax=ax, color="red", alpha=0.5)

    fig.savefig(output_path)


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
                label="Border",
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
    segmentation: SegmentationResults,
    bbox=None,
):
    angles = predictions.angles
    labels = dataset.labels
    segmentation = segmentation.segmentation.copy().astype(float)

    if bbox is not None:
        angles = angles[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        labels = labels[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        segmentation = segmentation[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        segmentation[segmentation == 0] = np.nan
        nuclei = labels > 0
        mask = labels == -1
    else:
        segmentation[segmentation == 0] = np.nan
        nuclei = labels > 0
        mask = labels == -1

    ax.imshow(nuclei.T, vmin=0, vmax=1, cmap="binary", interpolation="none")

    ax.imshow(
        segmentation.T,
        cmap="tab10",
        alpha=0.6,
        interpolation="none",
    )

    for xi in range(nuclei.shape[0]):
        for yi in range(nuclei.shape[1]):
            if mask[xi, yi]:
                dx, dy = pol2cart(0.5, angles[xi, yi])
                ax.arrow(xi + 0.5, yi + 0.5, dx, dy, width=0.07)
    ax.set_title("Predicted angles and segmentation")

    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=10,
                label="Nucleus",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Segmented Cell",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )


def plot_foreground(ax, predictions: ModelPredictions, bbox=None):
    foreground = predictions.foreground

    if bbox is not None:
        foreground = foreground[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    ax.imshow(foreground.T, vmin=0, vmax=1, cmap="coolwarm", interpolation="none")
    ax.set_title("Predicted foreground")


def update_projection(ax_dict, ax_key, projection="3d", fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = ax_dict[ax_key].get_subplotspec().get_geometry()
    ax_dict[ax_key].remove()
    ax_dict[ax_key] = fig.add_subplot(rows, cols, start + 1, projection=projection)


def plot_model_predictions(
    segmentation: SegmentationResults,
    dataset: Nuc2SegDataset,
    model_predictions: ModelPredictions,
    output_path=None,
    bbox=None,
):
    layout = """
    A
    B
    C
    """

    fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(10, 10))
    plot_angles_quiver(
        ax=ax["B"],
        dataset=dataset,
        predictions=model_predictions,
        segmentation=segmentation,
        bbox=bbox,
    )

    plot_labels(ax["A"], dataset, bbox=bbox)
    plot_foreground(ax["C"], model_predictions, bbox=bbox)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def plot_final_segmentation(nuclei_gdf, segmentation_gdf, output_path):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    ax.invert_yaxis()
    segmentation_gdf.plot(ax=ax, color="blue", edgecolor="lightgray", linewidth=0.1)
    nuclei_gdf.plot(ax=ax, color="red")

    fig.savefig(output_path)
    plt.close()


def plot_segmentation_comparison(
    seg_a, seg_b, nuclei, output_path, seg_a_name="sega", seg_b_name="segb", bbox=None
):
    if bbox:
        seg_a = seg_a[seg_a.geometry.within(bbox)]
        seg_b = seg_b[seg_b.geometry.within(bbox)]
        nuclei = nuclei[nuclei.geometry.within(bbox)]

    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    ax.invert_yaxis()
    seg_a.plot(ax=ax, color="blue", alpha=0.5, edgecolor="white", linewidth=0.5)
    seg_b.plot(ax=ax, color="red", alpha=0.5, edgecolor="white", linewidth=0.5)
    nuclei.plot(ax=ax, color="black", alpha=0.5, edgecolor="white", linewidth=0.5)

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
    plt.close()


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
    plt.close()


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
    plt.close()


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
        palette=cm["tab20"],
    )

    plt.tight_layout()
    fig.savefig(output_path)


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
            np.abs(aic_scores.min(axis=0) - aic_scores.min(axis=0)),
            np.abs(aic_scores.min(axis=0) - aic_scores.max(axis=0)),
        ]
    )

    # add min/max error bars
    ax.errorbar(
        n_components,
        aic_scores.min(axis=0),
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
        bic_scores.min(axis=0),
        yerr=error,
        label="BIC",
        alpha=0.5,
    )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Score")
    ax.set_title("AIC and BIC scores with error multiple trials")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "aic_bic_scores.pdf"))
    plt.close()

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
        plt.close()

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
        plt.close()

        # Create bar plot with the prior probabilities
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(range(len(final_prior_probs[idx])), final_prior_probs[idx])
        ax.set_xlabel("Cell type")
        ax.set_ylabel("Prior probability")
        ax.set_title(f"Prior probabilities for k={n_celltypes}")
        fig.savefig(os.path.join(output_dir, f"prior_probs_k={n_celltypes}.pdf"))
        fig.tight_layout()
        plt.close()


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
    ymin = np.Inf
    ymax = -np.Inf
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
