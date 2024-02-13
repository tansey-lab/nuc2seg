import geopandas
from shapely import box
from matplotlib import pyplot as plt
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
import numpy as np
import math


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


def plot_angles(ax, predictions: ModelPredictions, skip_factor=1, bbox=None):
    angles = predictions.angles

    if bbox is not None:
        angles = angles[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    skip = (slice(None, None, skip_factor), slice(None, None, skip_factor))

    ax.set_title("Predicted angles")

    U = 0.5 * np.cos(angles)
    V = 0.5 * np.sin(angles)
    Y, X = np.mgrid[0 : U.shape[0], 0 : U.shape[1]]

    ax.quiver(X[skip], Y[skip], U[skip], V[skip], color="black", headwidth=2)


def plot_foreground(ax, predictions: ModelPredictions, bbox=None):
    foreground = predictions.foreground

    if bbox is not None:
        foreground = foreground[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    ax.imshow(foreground, vmin=0, vmax=1, cmap="coolwarm", interpolation="none")
    ax.set_title("Predicted foreground")


def plot_model_predictions(
    dataset: Nuc2SegDataset,
    model_predictions: ModelPredictions,
    output_path=None,
    bbox=None,
):
    fig, ax = plt.subplots(figsize=(10, 10), nrows=3, dpi=1000)

    plot_labels(ax[0], dataset, bbox=bbox)
    plot_angles(ax[1], model_predictions, bbox=bbox)
    plot_foreground(ax[2], model_predictions, bbox=bbox)

    fig.tight_layout()
    fig.savefig(output_path)
