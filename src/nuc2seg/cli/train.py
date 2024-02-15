import argparse
import logging
import numpy as np

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, TiledDataset
from nuc2seg.unet_model import SparseUNet, Nuc2SegDataModule
from pytorch_lightning import Trainer


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a UNet model on preprocessed data."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--dataset",
        help="Path to dataset in h5 format.",
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
    parser.add_argument(
        "--n-devices",
        help="Number of devices to use for training.",
        type=int,
        default=1,
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
        "--n-filters",
        help="UNet hyperparameter.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-dataloader-workers",
        help="Number of workers to use for the data loader.",
        type=int,
        default=0,
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

    ds = Nuc2SegDataset.load_h5(args.dataset)

    tiled_ds = TiledDataset(
        ds,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
    )

    # Init DataModule
    dm = Nuc2SegDataModule(
        preprocessed_data_path=args.dataset,
        val_percent=args.val_percent,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
    )

    # Init model from datamodule's attributes
    model = SparseUNet(
        n_channels=600,
        n_classes=ds.n_classes,
        celltype_criterion_weights=tiled_ds.celltype_criterion_weights,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
        n_filters=args.n_filters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    # Init trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=args.n_devices,
        gradient_clip_val=args.gradient_clipping,
        gradient_clip_algorithm="norm",
    )

    # Fit model
    trainer.fit(model, dm)
