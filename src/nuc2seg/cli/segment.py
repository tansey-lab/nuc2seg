import argparse
import logging
import os.path
import numpy as np

from nuc2seg import log_config
from nuc2seg.segment import (
    greedy_cell_segmentation,
    convert_segmentation_to_shapefile,
    convert_transcripts_to_anndata,
)
from nuc2seg.data import Nuc2SegDataset, ModelPredictions, CelltypingResults
from nuc2seg.plotting import plot_final_segmentation, plot_segmentation_class_assignment
from nuc2seg.xenium import (
    read_transcripts_into_points,
    load_nuclei,
    create_shapely_rectangle,
)
from nuc2seg.celltyping import (
    predict_celltypes_for_segments_and_transcripts,
    select_best_celltyping_chain,
)

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
        "--transcripts",
        help="Xenium transcripts file in parquet format.",
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
        "--celltyping-results",
        help="Path to one or more celltyping results files.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--shapefile-output",
        help="Cell segmentation shapefile in parquet format.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--anndata-output",
        help="Cell-segmented anndata in h5ad format.",
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
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    return parser


def main():
    args = get_parser().parse_args()

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )

    else:
        sample_area = None

    transcripts = read_transcripts_into_points(args.transcripts)
    nuclei_gdf = load_nuclei(nuclei_file=args.nuclei_file, sample_area=sample_area)

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
    logger.info(f"Creating anndata")

    celltyping_chains = [CelltypingResults.load_h5(x) for x in args.celltyping_results]
    celltyping_results, aic_scores, bic_scores, best_k = select_best_celltyping_chain(
        celltyping_chains
    )

    celltype_predictions = predict_celltypes_for_segments_and_transcripts(
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        segment_geo_df=gdf,
        transcript_geo_df=transcripts,
        gene_names=celltyping_results.gene_names,
        max_distinace=0,
    )
    cell_type_labels = np.argmax(celltype_predictions, axis=1)
    gdf["celltype_assignment"] = cell_type_labels
    for i in range(celltype_predictions.shape[1]):
        gdf[f"celltype_{i}_prob"] = celltype_predictions[:, i]

    ad = convert_transcripts_to_anndata(
        transcript_gdf=transcripts, segmentation_gdf=gdf
    )

    logger.info(f"Saving anndata to {args.anndata_output}")
    ad.write_h5ad(args.anndata_output)

    logger.info(f"Plotting segmentation and class assignment.")
    plot_final_segmentation(
        nuclei_gdf=nuclei_gdf,
        segmentation_gdf=gdf,
        output_path=os.path.join(os.path.dirname(args.output), "segmentation.png"),
    )
    plot_segmentation_class_assignment(
        segmentation_gdf=gdf,
        output_path=os.path.join(os.path.dirname(args.output), "class_assignment.png"),
        cat_column="celltype_assignment",
    )

    logger.info(f"Saving shapefile to {args.shapefile_output}")
    gdf.to_parquet(args.shapefile_output)
