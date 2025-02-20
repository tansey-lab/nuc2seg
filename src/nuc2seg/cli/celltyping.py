import argparse
import logging
import anndata

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
        default=20,
    )
    parser.add_argument(
        "--min-n-celltypes",
        help="Minimum number of cell types to consider.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min-n-transcripts",
        help="Filter cells with less than this many transcripts from the training subset.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-cells",
        help="Maximum number of cells to sample from data for celltype estimation.",
        type=int,
        default=20_000,
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    seeds = np.random.SeedSequence(args.seed).spawn(args.n_chains)
    rng = np.random.default_rng(seeds[args.index])

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )

    else:
        sample_area = None

    adata = anndata.read_h5ad(args.adata)

    if sample_area:
        adata = filter_anndata_to_sample_area(adata, sample_area)

    adata = filter_anndata_to_min_transcripts(adata, args.min_n_transcripts)

    adata = subset_anndata(adata, args.max_cells, rng=rng)

    celltype_results = fit_celltyping_on_adata(
        adata=adata,
        min_components=args.min_n_celltypes,
        max_components=args.max_n_celltypes,
        rng=rng,
    )

    celltype_results.save_h5(args.output)
