import argparse
import logging
import os.path

import geopandas

from nuc2seg import log_config
from nuc2seg.data import (
    Nuc2SegDataset,
    ModelPredictions,
)
from nuc2seg.utils import create_shapely_rectangle
from nuc2seg.plotting import plot_model_predictions

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot raw UNet predictions vs segmentation vs raw data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--predictions",
        help="Model prediction output in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prior-segmentation",
        help="Prior segmentation in geoparquet.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segmentation",
        help="Segmentation in geoparquet.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Preprocessed dataset in h5 format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for plots.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )
    else:
        sample_area = None

    log_config.configure_logging(args)

    logger.info(f"Loading model predictions from {args.predictions}")
    predictions = ModelPredictions.load_h5(args.predictions)

    logger.info(f"Loading prior segmentation from {args.prior_segmentation}")
    prior_segmentation = geopandas.read_parquet(args.prior_segmentation)

    logger.info(f"Loading segmentation from {args.segmentation}")
    segmentation = geopandas.read_parquet(args.segmentation)

    logger.info(f"Loading dataset from {args.dataset}")
    dataset = Nuc2SegDataset.load_h5(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_model_predictions(
        model_predictions=predictions,
        dataset=dataset,
        prior_segmentation_gdf=prior_segmentation,
        segmentation_gdf=segmentation,
        bbox=sample_area,
        output_path=os.path.join(
            args.output_dir, "_".join([str(x) for x in sample_area.bounds]) + ".pdf"
        ),
    )
