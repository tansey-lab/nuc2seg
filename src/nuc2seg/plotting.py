import geopandas
from shapely import box
from matplotlib import pyplot as plt
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
import numpy as np
import math
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
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
    fig, ax = plt.subplots(figsize=(15, 15))
    segmentation_gdf.plot(
        ax=ax,
        color="blue",
    )
    nuclei_gdf.plot(ax=ax, color="red")

    fig.savefig(output_path)
    plt.close()


def plot_segmentation_class_assignment(segmentation_gdf, output_path):
    fig, ax = plt.subplots(figsize=(15, 15))
    segmentation_gdf.plot(
        ax=ax,
        categorical=True,
        column="class_assignment",
        legend=True,
        cmap="tab20",
    )
    fig.savefig(output_path)
    plt.close()
