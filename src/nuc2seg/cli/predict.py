import argparse
import logging
import os.path

from pytorch_lightning import Trainer

from nuc2seg import log_config
from nuc2seg.segment import forward_pass_result_to_obj
from nuc2seg.unet_model import SparseUNet, Nuc2SegDataModule

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a UNet model on preprocessed data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Model prediction output dir.",
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
        "--model-weights",
        help="File to read model weights from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tile-height",
        help="Height of the tiles.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--tile-width",
        help="Width of the tiles.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--overlap-percentage",
        help="What percent of each tile dimension overlaps with the next tile.",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--num-dataloader-workers",
        help="Number of workers to use for the data loader.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
        help="Device to use for prediction.",
        type=str,
        default="auto",
        choices=["cpu", "gpu", "tpu", "ipu", "mps", "auto"],
    )
    parser.add_argument(
        "--n-jobs",
        help="Number of jobs to use for parallel prediction",
        type=int,
        default=None,
    )
    parser.add_argument("--job-index", help="current job index", type=int, default=None)
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    logger.info(f"Loading dataset from {args.dataset}")

    model = SparseUNet.load_from_checkpoint(args.model_weights)

    dm = Nuc2SegDataModule(
        preprocessed_data_path=args.dataset,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
        num_workers=args.num_dataloader_workers,
        predict_n_threads=args.n_jobs,
        predict_thread_idx=args.job_index,
    )

    trainer = Trainer(
        accelerator=args.device,
        default_root_dir=os.path.dirname(args.model_weights),
    )

    logger.info(f"Writing output to {args.dataset}")

    for item in trainer.predict(model, dm):
        value = item["value"]
        tile_index = item["tile_index"]
        model_prediction = forward_pass_result_to_obj(value)
        model_prediction.save_h5(
            os.path.join(args.output_dir, f"model_prediction_tile_{tile_index}.h5")
        )
