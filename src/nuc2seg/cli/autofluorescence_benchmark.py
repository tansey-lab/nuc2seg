import argparse
import logging
import os.path

import nuc2seg.post_xenium_imaging
from nuc2seg import benchmark
from nuc2seg import log_config
from nuc2seg import plotting
from nuc2seg.data import SegmentationResults
from nuc2seg.xenium import read_xenium_cell_segmentation_masks

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark cell segmentation given post-Xenium IF data that includes an autofluorescence marker."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Folder for output plots.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--xenium-cells",
        help="Xenium cell segmentation in zarr format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segmentation",
        help="Nuc2Seg segmentation in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--if-ome-tiff",
        help="Post xenium immunoflorescence ometiff file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--morphology-ome-tiff",
        help="Xenium morphology ometiff file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--alignment-matrix",
        help="Affine transformation matrix in CSV format (generated by xenium browser).",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--if-channel-name",
        help="IF channel name (from ometiff metadata) to use for benchmarking.",
        type=str,
        required=True,
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    logger.info(f"Loading segmentation from {args.segmentation}")
    segmentation = SegmentationResults.load_h5(args.segmentation)

    logger.info(
        f"Loading IF data from {args.if_ome_tiff} and {args.morphology_ome_tiff}"
    )
    channel_names, aligned_if_intensities = (
        nuc2seg.post_xenium_imaging.load_immunofluorescence(
            segmentation=segmentation,
            if_ome_tiff_path=args.if_ome_tiff,
            morphology_mip_ome_tiff_path=args.morphology_ome_tiff,
            alignment_matrix_path=args.alignment_matrix,
        )
    )
    channel_index = channel_names.index(args.if_channel_name)

    target_if_intensities = aligned_if_intensities[channel_index, ...]
    del aligned_if_intensities

    logger.info(f"Loading predictions from {args.predictions}")

    logger.info(f"Loading xenium cell segmentation from {args.xenium_cells}")
    xenium_cell_segmentation_mask = read_xenium_cell_segmentation_masks(
        args.xenium_cells,
        x_extent_pixels=segmentation.segmentation.shape[0],
        y_extent_pixels=segmentation.segmentation.shape[1],
    )

    foreground_intensities, background_intensities = (
        benchmark.score_foreground_segmentation(
            (segmentation.segmentation > 0), target_if_intensities
        )
    )

    xenium_foreground_intensities, xenium_background_intensities = (
        benchmark.score_foreground_segmentation(
            (xenium_cell_segmentation_mask > 0).astype(int), target_if_intensities
        )
    )

    plotting.plot_foreground_background_benchmark(
        foreground_intensities,
        background_intensities,
        os.path.join(args.output_dir, "nuc2seg_foreground_background_benchmark.png"),
    )

    plotting.plot_foreground_background_benchmark(
        xenium_foreground_intensities,
        xenium_background_intensities,
        os.path.join(args.output_dir, "xenium_foreground_background_benchmark.png"),
    )

    logger.info(f"Creating boxplot")
    plotting.foreground_background_boxplot(
        nuc2seg_foreground_intensities=foreground_intensities,
        nuc2seg_background_intensities=background_intensities,
        other_method_foreground_intensities=xenium_foreground_intensities,
        other_method_background_intensities=xenium_background_intensities,
        output_path=os.path.join(args.output_dir, "foreground_background_boxplot.png"),
        other_method_name="Xenium",
    )

    nuc2seg_intensities, nuc2seg_segment_sizes = (
        benchmark.get_per_segment_immunofluorescence_intensity(
            segmentation=segmentation.segmentation,
            immunofluorescence=target_if_intensities,
        )
    )

    xenium_intensities, xenium_segment_sizes = (
        benchmark.get_per_segment_immunofluorescence_intensity(
            segmentation=xenium_cell_segmentation_mask,
            immunofluorescence=target_if_intensities,
        )
    )

    plotting.plot_segmentation_avg_intensity_distribution(
        nuc2seg_intensities,
        xenium_intensities,
        os.path.join(args.output_dir, "segmentation_avg_intensity_distribution.png"),
        other_method_name="Xenium",
    )

    plotting.plot_segmentation_size_distribution(
        nuc2seg_segment_sizes,
        xenium_segment_sizes,
        os.path.join(args.output_dir, "segmentation_size_distribution.png"),
        other_method_name="Xenium",
    )
