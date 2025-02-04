import argparse
import logging
import os.path

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, TiledDataset, TrainTestSplit
from nuc2seg.unet_model import SparseUNet, Nuc2SegDataModule

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
        "--checkpoint",
        help="Path to a checkpoint to continue training.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save model checkpoints to.",
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
        default=30,
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
        "--betas", help="Betas.", type=float, default=(0.9, 0.999), nargs=2
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
        default="auto",
        choices=["cpu", "gpu", "tpu", "ipu", "mps", "auto"],
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
    parser.add_argument(
        "--val-check-interval",
        help="Check validation set after this many fractional epochs.",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--foreground-loss-factor",
        help="Multiply foreground loss by this factor before backpropagation.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--unlabeled-foreground-loss-factor",
        help="Multiply unlabeled foreground loss by this factor before backpropagation.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--celltype-loss-factor",
        help="Multiply celltype loss by this factor before backpropagation.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--angle-loss-factor",
        help="Multiply angle loss by this factor before backpropagation.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--loss-reweighting",
        help="Reweight losses to be even.",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    celltype_frequencies = ds.get_celltype_frequencies()
    background_frequencies = ds.get_background_frequencies()

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
        train_batch_size=1,
        val_batch_size=1,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        tile_overlap=args.overlap_percentage,
        num_workers=args.num_dataloader_workers,
    )

    if args.checkpoint is None:
        logger.info("Training new model")
        model = SparseUNet(
            n_channels=ds.n_genes,
            n_classes=ds.n_classes,
            angle_loss_factor=args.angle_loss_factor,
            foreground_loss_factor=args.foreground_loss_factor,
            unlabeled_foreground_loss_factor=args.unlabeled_foreground_loss_factor,
            celltype_loss_factor=args.celltype_loss_factor,
            celltype_criterion_weights=tiled_ds.celltype_criterion_weights,
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            tile_overlap=args.overlap_percentage,
            n_filters=args.n_filters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=args.betas,
            loss_reweighting=args.loss_reweighting,
            celltype_frequencies=celltype_frequencies,
            background_frequencies=background_frequencies,
        )
    else:
        logger.info(f"Resuming from checkpoint {args.checkpoint}")
        model = SparseUNet.load_from_checkpoint(
            args.checkpoint,
            angle_loss_factor=args.angle_loss_factor,
            foreground_loss_factor=args.foreground_loss_factor,
            celltype_loss_factor=args.celltype_loss_factor,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=args.betas,
            loss_reweighting=args.loss_reweighting,
        )

    # save checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_accuracy",
        mode="max",
    )

    # Init trainer
    wandb_logger = WandbLogger(
        log_model=True,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=args.n_devices,
        gradient_clip_val=args.gradient_clipping,
        gradient_clip_algorithm="norm",
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_callback],
        log_every_n_steps=min(50, len(tiled_ds) - 1),
    )

    # Fit model
    trainer.fit(model, dm)

    test_indices = np.array([x for x in dm.val_set.indices]).astype(int)
    train_indices = np.array([x for x in dm.train_set.indices]).astype(int)

    TrainTestSplit(
        train_indices=test_indices,
        test_indices=train_indices,
    ).save_h5(os.path.join(args.output_dir + "/train_val_test_assignments.h5"))
