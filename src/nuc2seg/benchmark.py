import numpy as np
import logging
import tqdm

logger = logging.getLogger(__name__)


def get_per_segment_immunofluorescence_intensity(
    segmentation: np.ndarray, immunofluorescence: np.ndarray, maximum_cells=4000
):

    intensities = []
    segment_sizes = []

    for value in tqdm.tqdm(np.unique(segmentation)[:maximum_cells]):
        if value in [-1, 0]:
            continue
        mask = segmentation == value
        size = np.count_nonzero(mask)
        mean_intensity = immunofluorescence[mask].mean()
        intensities.append(mean_intensity)
        segment_sizes.append(size)

    return intensities, segment_sizes


def score_foreground_segmentation(
    predictions: np.ndarray,
    immunofluorescence: np.ndarray,
):
    foreground_mask = predictions > 0.5
    background_mask = predictions <= 0.5

    foreground_intensity = immunofluorescence[foreground_mask]
    background_intensity = immunofluorescence[background_mask]

    return foreground_intensity, background_intensity
