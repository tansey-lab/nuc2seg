import geopandas
import pandas
from shapely import box
from matplotlib import pyplot as plt
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
import numpy as np
import math
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from nuc2seg.preprocessing import pol2cart
import os.path


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


def plot_angles(ax, predictions: ModelPredictions, bbox=None):

    angles = predictions.angles

    if bbox is not None:
        angles = angles[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    ax.imshow(angles.T, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    ax.set_title("Predicted angles")


def plot_angles_quiver(
    ax, dataset: Nuc2SegDataset, predictions: ModelPredictions, bbox=None
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

    ax.imshow(nuclei.T, vmin=0, vmax=1, cmap="binary", interpolation="none")

    for xi in range(nuclei.shape[0]):
        for yi in range(nuclei.shape[1]):
            if mask[xi, yi]:
                dx, dy = pol2cart(0.5, angles[xi, yi])
                ax.arrow(xi + 0.5, yi + 0.5, dx, dy, width=0.07, alpha=0.5)
    ax.set_title("Predicted angles")


def plot_angle_legend(ax):
    # Define colormap normalization for 0 to 2*pi
    norm = Normalize(-np.pi, np.pi)

    # Plot a color mesh on the polar plot
    # with the color set by the angle

    n = 200  # the number of secants for the mesh
    t = np.linspace(-np.pi, np.pi, n)  # theta values
    r = np.linspace(0.8, 1, 2)  # radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
    c = tg  # define color values as theta value
    im = ax.pcolormesh(
        t, r, c.T, norm=norm, cmap="hsv"
    )  # plot the colormesh on axis with colormap
    ax.set_theta_direction(-1)
    # remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Angle Legend", fontsize=9)  # title with padding for y label
    # decrease left margin
    plt.subplots_adjust(left=-0.5)
    ax.spines["polar"].set_visible(False)  # turn off the axis spine.


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
    dataset: Nuc2SegDataset,
    model_predictions: ModelPredictions,
    output_path=None,
    bbox=None,
    use_quiver=True,
):

    if use_quiver:
        layout = """
        A
        B
        C
        """
    else:
        layout = """
        A.
        BD
        C.
        """

    if use_quiver:
        fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(10, 10))
        plot_angles_quiver(ax["B"], dataset, model_predictions, bbox=bbox)
    else:
        fig, ax = plt.subplot_mosaic(
            mosaic=layout, figsize=(10, 10), width_ratios=[9, 1]
        )
        plot_angles(ax["B"], model_predictions, bbox=bbox)
        update_projection(ax, "D", projection="polar", fig=fig)
        plot_angle_legend(ax["D"])

    plot_labels(ax["A"], dataset, bbox=bbox)
    plot_foreground(ax["C"], model_predictions, bbox=bbox)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def plot_final_segmentation(nuclei_gdf, segmentation_gdf, output_path):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    segmentation_gdf.plot(
        ax=ax,
        color="blue",
    )
    nuclei_gdf.plot(ax=ax, color="red")

    fig.savefig(output_path)
    plt.close()


def plot_segmentation_class_assignment(segmentation_gdf, output_path):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=1000)
    segmentation_gdf.plot(
        ax=ax,
        categorical=True,
        column="class_assignment",
        legend=True,
        cmap="tab20",
    )
    fig.savefig(output_path)
    plt.close()


def plot_celltype_estimation_results(
    aic_scores,
    bic_scores,
    final_expression_profiles,
    final_prior_probs,
    final_cell_types,
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
    nuc2seg_intensities, other_intensities, output_path
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(nuc2seg_intensities, bins=100, alpha=0.5, label="Nuc2Seg")
    ax.hist(other_intensities, bins=100, alpha=0.5, label="Xenium")
    ax.set_xlabel("Avg Intensity of Cell Segment")
    ax.set_ylabel("Frequency")
    ax.set_title("Average segment intensity distributions")
    ax.legend()

    fig.savefig(output_path)


def plot_segmentation_size_distribution(
    nuc2seg_intensities, other_intensities, output_path
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(nuc2seg_intensities, bins=100, alpha=0.5, label="Nuc2Seg")
    ax.hist(other_intensities, bins=100, alpha=0.5, label="Xenium")
    ax.set_xlabel("Segment # of Pixels")
    ax.set_ylabel("Frequency")
    ax.set_title("Average segment size distributions")
    ax.legend()

    fig.savefig(output_path)


def foreground_background_boxplot(
    nuc2seg_foreground_intensities,
    nuc2seg_background_intensities,
    xenium_foreground_intensities,
    xenium_background_intensities,
    output_path,
):
    import seaborn as sns

    sns.set_theme(style="ticks", palette="pastel")

    df = pandas.DataFrame()
    df["intensity"] = np.concatenate(
        [
            nuc2seg_foreground_intensities,
            nuc2seg_background_intensities,
            xenium_foreground_intensities,
            xenium_background_intensities,
        ]
    )

    df["method"] = (
        ["Nuc2Seg"] * len(nuc2seg_foreground_intensities)
        + ["Nuc2Seg"] * len(nuc2seg_background_intensities)
        + ["Xenium"] * len(xenium_foreground_intensities)
        + ["Xenium"] * len(xenium_background_intensities)
    )
    df["class"] = (
        ["Foreground"] * len(nuc2seg_foreground_intensities)
        + ["Background"] * len(nuc2seg_background_intensities)
        + ["Foreground"] * len(xenium_foreground_intensities)
        + ["Background"] * len(xenium_background_intensities)
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
