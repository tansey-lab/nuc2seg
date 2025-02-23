import argparse
import logging
import os.path

from matplotlib import pyplot as plt

from nuc2seg import log_config
from nuc2seg.celltyping import (
    select_best_celltyping_chain,
)
from nuc2seg.data import CelltypingResults
from nuc2seg.plotting import plot_celltype_estimation_results, rank_genes_groups_plot

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for plotting the results of celltype estimation."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Output directory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--celltyping-results",
        help="Path to one or more celltyping results files.",
        type=str,
        required=True,
        nargs="+",
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    celltyping_chains = [CelltypingResults.load_h5(x) for x in args.celltyping_results]
    if args.n_celltypes:
        best_k = list(celltyping_chains[0].n_component_values).index(args.n_celltypes)
        celltyping_results, aic_scores, bic_scores, best_k = (
            select_best_celltyping_chain(celltyping_chains, best_k)
        )
    else:
        celltyping_results, aic_scores, bic_scores, best_k = (
            select_best_celltyping_chain(celltyping_chains, None)
        )

    logger.info("Plotting celltype estimation results")

    plot_celltype_estimation_results(
        aic_scores,
        bic_scores,
        celltyping_results.expression_profiles,
        celltyping_results.prior_probs,
        celltyping_results.relative_expression,
        celltyping_results.n_component_values,
        os.path.join(os.path.dirname(args.output_dir), "cell_typing_plots"),
    )

    logger.info("Plotting rank_genes_groups_plot")
    for k in celltyping_results.n_component_values:
        rank_genes_groups_plot(
            celltyping_results=celltyping_results,
            k=k,
            output_path=os.path.join(
                os.path.join(os.path.dirname(args.output_dir), "cell_typing_plots"),
                f"rank_genes_groups_k={k}.pdf",
            ),
            n_genes=10,
            sharey=False,
        )
        plt.close()
        rank_genes_groups_plot(
            celltyping_results=celltyping_results,
            k=k,
            output_path=os.path.join(
                os.path.join(os.path.dirname(args.output_dir), "cell_typing_plots"),
                f"rank_genes_groups_sharey_k={k}.pdf",
            ),
            n_genes=10,
            sharey=True,
        )
        plt.close()
    logger.info("Finished plotting rank_genes_groups_plot")
