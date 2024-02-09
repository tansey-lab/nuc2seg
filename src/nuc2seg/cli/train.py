import argparse
import logging
import numpy as np
import torch

from nuc2seg import log_config
from nuc2seg.train import train
from nuc2seg.unet_model import SparseUNet

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a UNet model on preprocessed data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--preprocessed-tiles-dir",
        help="Directory containing preprocessed tiles.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-weights-output",
        help="File to save model weights to.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-classes",
        help="Number of classes to segment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        help="Seed to use for PRNG.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs to train for.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate.",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--val-percent",
        help="Percentage of data to use for validation.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--save-checkpoint",
        help="Save model checkpoint.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--amp",
        help="Use automatic mixed precision.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--weight-decay",
        help="Weight decay.",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--momentum",
        help="Momentum.",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--gradient-clipping",
        help="Gradient clipping.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--validation-frequency",
        help="Frequency of validation.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-workers",
        help="Maximum number of workers to use for data loading.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        help="Device to use for training.",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    np.random.seed(args.seed)
    model = SparseUNet(600, args.n_classes + 2, (64, 64))

    train(
        model,
        device=args.device,
        tiles_dir=args.preprocessed_tiles_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_percent=args.val_percent,
        save_checkpoint=args.save_checkpoint,
        amp=args.amp,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        gradient_clipping=args.gradient_clipping,
        max_workers=args.max_workers,
        validation_frequency=args.validation_frequency,
    )

    logger.info(f"Saving model weights to {args.model_weights_output}")
    torch.save(model.state_dict(), args.model_weights_output)
