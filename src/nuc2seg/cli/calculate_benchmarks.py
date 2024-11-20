import argparse
import logging

import numpy as np
import torch
import tqdm
import os

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, TiledDataset
from nuc2seg.segment import stitch_predictions
from nuc2seg.xenium import load_vertex_file, load_and_filter_transcripts_as_points
from nuc2seg.postprocess import (
    calculate_average_intersection_over_union,
    calculate_segmentation_jaccard_index,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compare multiple segmentation results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Directory for output tables and plots.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segmentation-files",
        help="Segmentation files in GeoParquet format from each method.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--segmentation-method-names",
        help="Segmentation method names "
        "(in the same order as the entries in `--segmentation-files`).",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--true-boundaries",
        help="Ground truth segmentation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--nuclei-boundaries",
        help="Nuclei segmentations.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--transcripts",
        help="Transcripts in parquet.",
        type=str,
        required=True,
    )
    return parser


def main():
    args = get_parser().parse_args()

    logger.info("Loading true boundaries.")
    true_boundaries = load_vertex_file(args.true_boundaries)

    logger.info("Loading transcripts.")
    transcripts = load_and_filter_transcripts_as_points(args.transcripts)

    segmentations = {}
    ious = {}
    jaccards = {}

    os.makedirs(args.output_dir, exist_ok=True)

    for method_name, seg_fn in zip(
        args.segmentation_method_names, args.segmentation_files
    ):
        logger.info(f"Loading segmentation from {seg_fn}.")
        segmentation_shapes = load_vertex_file(seg_fn)

        logger.info(f"Calculating benchmarks for {method_name}.")
        iou = calculate_average_intersection_over_union(
            segmentation_shapes, true_boundaries
        )
        jaccard = calculate_segmentation_jaccard_index(
            transcripts, segmentation_shapes, true_boundaries
        )

        segmentations[method_name] = segmentation_shapes
        ious[method_name] = iou
        jaccards[method_name] = jaccard

        iou.to_parquet(f"{args.output_dir}/{method_name}_iou.parquet")
        jaccard.to_parquet(f"{args.output_dir}/{method_name}_jaccard.parquet")
