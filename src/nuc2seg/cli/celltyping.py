import argparse
import logging
import anndata
import torch
import numpy as np

from nuc2seg import log_config
from nuc2seg.celltyping import fit_celltyping_on_adata
from nuc2seg.utils import (
    create_shapely_rectangle,
    filter_anndata_to_sample_area,
    filter_anndata_to_min_transcripts,
    subset_anndata,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run celltype estimation on nucelus-segmented reads."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output",
        help="Output path.",
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
        "--index",
        help="Chain index.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n-chains",
        help="Number of parallel chains with different random initialization for cell typing.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--adata",
        default=None,
        type=str,
        help="Anndata containing nuclear segmented data.",
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--max-n-celltypes",
        help="Maximum number of cell types to consider (inclusive).",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--min-n-celltypes",
        help="Minimum number of cell types to consider.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-cells",
        help="Maximum number of cells to sample from data for celltype estimation.",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--transcript-count-percentile",
        help="Only use cells above this percentile of total transcript count for model fitting",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--device",
        help="Device to use for training.",
        type=str,
        default="auto",
        choices=["cpu", "gpu", "tpu", "ipu", "mps", "auto"],
    )
    return parser


def get_device(device: str):
    if device == "auto":
        if torch.cuda.is_available():
            return "gpu"
        else:
            return "cpu"
    else:
        return device


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    log_config.configure_logging(args)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    seeds = np.random.SeedSequence(args.seed).spawn(args.n_chains)
    rng = np.random.default_rng(seeds[args.index])
    seed = rng.integers(0, 2**32)

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )

    else:
        sample_area = None

    adata = anndata.read_h5ad(args.adata)

    if sample_area:
        adata = filter_anndata_to_sample_area(adata, sample_area)

    total_per_cell = np.array(adata.X.sum(axis=1)).squeeze()
    cutoff = np.floor(np.percentile(total_per_cell, 25))

    adata = filter_anndata_to_min_transcripts(adata, min_transcripts=cutoff)

    adata = subset_anndata(adata, args.max_cells, rng=rng)

    celltype_results = fit_celltyping_on_adata(
        adata=adata,
        min_components=args.min_n_celltypes,
        max_components=args.max_n_celltypes,
        seed=seed,
        device=device,
    )

    celltype_results.save_h5(args.output)
