import pandas
import pathlib
from pyometiff import OMETIFFReader
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_if(if_ome_tiff_path, morphology_mip_ome_tiff_path, alignment_matrix_path):
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
        res = cv2.warpAffine(
            src=src, dst=dst, M=M[:2, :], dsize=(dst.shape[1], dst.shape[0])
        )
        transformed_images.append(res)

    return channel_names_in_order, np.stack(transformed_images)
