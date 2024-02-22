import argparse
import logging
import os.path

from nuc2seg import log_config
from nuc2seg.segment import greedy_cell_segmentation, convert_segmentation_to_shapefile
from nuc2seg.data import Nuc2SegDataset, ModelPredictions
from nuc2seg.plotting import plot_final_segmentation, plot_segmentation_class_assignment

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Segment cell nuclei.")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output",
        help="Cell segmentation output in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--shapefile-output",
        help="Cell segmentation shapefile in parquet format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        help="Model predictions in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-steps",
        help="Maximum number of iterations for greedy expansion algorithm, may terminate early if "
        "all non-background pixels are assigned to a cell.",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--foreground-prob-threshold",
        help="Threshold for considering pixel foreground, number between 0-1.",
        type=float,
        default=0.5,
    )
    return parser


def main():
    args = get_parser().parse_args()

    dataset = Nuc2SegDataset.load_h5(args.dataset)
    predictions = ModelPredictions.load_h5(args.predictions)

    result = greedy_cell_segmentation(
        dataset=dataset,
        predictions=predictions,
        max_expansion_steps=args.max_steps,
        foreground_threshold=args.foreground_prob_threshold,
    )

    logger.info(f"Saving segmentation to {args.output}")
    result.save_h5(args.output)

    gdf = convert_segmentation_to_shapefile(
        segmentation=result.segmentation, dataset=dataset, predictions=predictions
    )

    nuclei_gdf = convert_segmentation_to_shapefile(
        segmentation=dataset.labels, dataset=dataset, predictions=predictions
    )

    plot_final_segmentation(
        nuclei_gdf=nuclei_gdf,
        segmentation_gdf=gdf,
        output_path=os.path.join(os.path.dirname(args.output), "segmentation.png"),
    )
    plot_segmentation_class_assignment(
        segmentation_gdf=gdf,
        output_path=os.path.join(os.path.dirname(args.output), "class_assignment.png"),
    )
    gdf.to_parquet(args.shapefile_output)
