import geopandas
from shapely import box
from matplotlib import pyplot as plt
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
import numpy as np
import math
from matplotlib.colors import Normalize


def plot_tiling(bboxes, output_path):
    polygons = [box(*x) for x in bboxes]
    gdf = geopandas.GeoDataFrame(geometry=polygons)

    fig, ax = plt.subplots()

    gdf.boundary.plot(ax=ax, color="red", alpha=0.5)

    fig.savefig(output_path)


def plot_labels(ax, dataset: Nuc2SegDataset, bbox=None):
    label_plot = dataset.labels.copy()
    transcripts = dataset.transcripts.copy()
    label_plot[label_plot >= 1] = 5
    label_plot[label_plot == -1] = 2

    if bbox is not None:
        label_plot = label_plot[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask_x = (transcripts[:, 0] >= bbox[0]) & (transcripts[:, 0] < bbox[2])
        mask_y = (transcripts[:, 1] >= bbox[1]) & (transcripts[:, 1] < bbox[3])
        mask = mask_x & mask_y
        transcripts = transcripts[mask]

        transcripts[:, 0] = transcripts[:, 0] - bbox[0]
        transcripts[:, 1] = transcripts[:, 1] - bbox[1]

    ax.set_title("Labels and transcripts")
    ax.imshow(label_plot, cmap="copper", interpolation="none")

    ax.scatter(
        transcripts[:, 1], transcripts[:, 0], color="red", zorder=100, s=0.1, alpha=0.3
    )


def plot_angles(ax, predictions: ModelPredictions, bbox=None):

    angles = predictions.angles

    if bbox is not None:
        angles = angles[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    ax.imshow(angles, vmin=-np.pi, vmax=np.pi, cmap="hsv")
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

    ax.imshow(foreground, vmin=0, vmax=1, cmap="coolwarm", interpolation="none")
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
):

    layout = """
    A.
    BD
    C.
    """

    fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(10, 10), width_ratios=[9, 1])

    plot_labels(ax["A"], dataset, bbox=bbox)
    plot_angles(ax["B"], model_predictions, bbox=bbox)
    plot_foreground(ax["C"], model_predictions, bbox=bbox)

    update_projection(ax, "D", projection="polar", fig=fig)

    plot_angle_legend(ax["D"])

    fig.tight_layout()
    fig.savefig(output_path)
