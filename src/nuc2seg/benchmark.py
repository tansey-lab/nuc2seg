import pandas
import pathlib
from pyometiff import OMETIFFReader
import cv2
import numpy as np
import logging
import tqdm

from nuc2seg.data import SegmentationResults, ModelPredictions

logger = logging.getLogger(__name__)


def load_immunofluorescence(
    segmentation: SegmentationResults,
    if_ome_tiff_path,
    morphology_mip_ome_tiff_path,
    alignment_matrix_path,
):
    M = pandas.read_csv(alignment_matrix_path, header=None).values

    reader = OMETIFFReader(fpath=pathlib.Path(if_ome_tiff_path))
    if_img_array, if_metadata, if_xml_metadata = reader.read()

    channel_names_in_order = [
        y["Name"]
        for y in sorted(list(if_metadata["Channels"].values()), key=lambda x: x["ID"])
    ]

    logger.info(
        f"File {if_ome_tiff_path} has the following channels: {channel_names_in_order}"
    )

    reader = OMETIFFReader(fpath=pathlib.Path(morphology_mip_ome_tiff_path))
    xenium_img_array, xenium_metadata, xenium_xml_metadata = reader.read()

    if len(xenium_img_array.shape) == 3:
        xenium_img_array = xenium_img_array[0, ...]

    transformed_images = []

    for i in range(0, if_img_array.shape[0]):
        src = if_img_array[i, ...]
        dst = np.zeros_like(xenium_img_array)

        # Warp the IF image so it lines up with the xenium coordinates
        res = cv2.warpAffine(
            src=src, dst=dst, M=M[:2, :], dsize=(dst.shape[1], dst.shape[0])
        ).T

        # At this point we are aligned,
        # and we just need to down-sample down to the same resolution as the segmentation
        resized = cv2.resize(
            res,
            (segmentation.segmentation.shape[0], segmentation.segmentation.shape[1]),
        ).T
        transformed_images.append(resized)

    return channel_names_in_order, np.stack(transformed_images)


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
