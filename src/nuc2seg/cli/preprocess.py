import argparse
import logging
import os.path

import pandas
from matplotlib import pyplot as plt

from nuc2seg import log_config
from nuc2seg.celltyping import (
    select_best_celltyping_chain,
    predict_celltypes_for_segments_and_transcripts,
)
from nuc2seg.data import CelltypingResults
from nuc2seg.plotting import plot_celltype_estimation_results, rank_genes_groups_plot
from nuc2seg.preprocessing import create_rasterized_dataset, create_nuc2seg_dataset
from nuc2seg.xenium import (
    load_vertex_file,
    load_and_filter_transcripts_as_points,
)
from nuc2seg.utils import create_shapely_rectangle

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="This is a utility for preprocessing Xenium data for the model."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--transcripts-file",
        help="Path to the Xenium transcripts parquet file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--nuclei-file",
        help="Path to the Xenium nuclei boundaries parquet file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output path.",
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
    parser.add_argument(
        "--resolution",
        help="Size of a pixel in microns for rasterization.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-qv",
        help="Minimum quality value for a transcript to be included.",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--foreground-nucleus-distance",
        help="Distance from a nucleus to be considered foreground.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--background-nucleus-distance",
        help="Distance from a nucleus to be considered background.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--background-transcript-distance",
        help="Distance from a transcript to be considered background.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--background-pixel-transcripts",
        help="Number of transcripts in a pixel to be considered background.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--n-celltypes",
        default=None,
        type=str,
        help="Force using this number of celltypes, otherwise pick via BIC.",
    )
    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )

    else:
        df = pandas.read_parquet(
            args.transcripts_file, columns=["x_location", "y_location"]
        )
        y_max = df["y_location"].max()
        x_max = df["x_location"].max()
        del df

        sample_area = create_shapely_rectangle(0, 0, x_max, y_max)

    nuclei_geo_df = load_vertex_file(
        fn=args.nuclei_file,
        sample_area=sample_area,
    )

    tx_geo_df = load_and_filter_transcripts_as_points(
        transcripts_file=args.transcripts_file,
        sample_area=sample_area,
        min_qv=args.min_qv,
    )

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

    logger.info("Predicting celltypes for segments and transcripts")

    celltype_predictions = predict_celltypes_for_segments_and_transcripts(
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        segment_geo_df=nuclei_geo_df,
        transcript_geo_df=tx_geo_df,
        max_distinace=args.foreground_nucleus_distance,
    )

    logger.info("Plotting celltype estimation results")

    plot_celltype_estimation_results(
        aic_scores,
        bic_scores,
        celltyping_results.expression_profiles,
        celltyping_results.prior_probs,
        celltyping_results.relative_expression,
        celltyping_results.n_component_values,
        os.path.join(os.path.dirname(args.output), "cell_typing_plots"),
    )

    logger.info("Plotting rank_genes_groups_plot")
    for k in celltyping_results.n_component_values:
        rank_genes_groups_plot(
            celltyping_results=celltyping_results,
            k=k,
            output_path=os.path.join(
                os.path.join(os.path.dirname(args.output), "cell_typing_plots"),
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
                os.path.join(os.path.dirname(args.output), "cell_typing_plots"),
                f"rank_genes_groups_sharey_k={k}.pdf",
            ),
            n_genes=10,
            sharey=True,
        )
        plt.close()

    rasterized_dataset = create_rasterized_dataset(
        nuclei_geo_df=nuclei_geo_df,
        tx_geo_df=tx_geo_df,
        sample_area=sample_area,
        resolution=args.resolution,
        foreground_nucleus_distance=args.foreground_nucleus_distance,
        background_nucleus_distance=args.background_nucleus_distance,
        background_pixel_transcripts=args.background_pixel_transcripts,
        background_transcript_distance=args.background_transcript_distance,
    )

    del tx_geo_df
    del nuclei_geo_df

    logger.info("Creating Nuc2Seg dataset")

    ds = create_nuc2seg_dataset(rasterized_dataset, celltype_predictions)

    logger.info("Saving to h5")
    ds.save_h5(args.output)
