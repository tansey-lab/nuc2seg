import argparse
import logging

import numpy as np
import pandas
import h5py

from nuc2seg import log_config
from nuc2seg.celltyping import (
    predict_celltypes_for_anndata,
    select_best_celltyping_chain,
)
from nuc2seg.data import Nuc2SegDataset, ModelPredictions, CelltypingResults
from nuc2seg.segment import (
    greedy_cell_segmentation,
    convert_segmentation_to_shapefile,
)
from nuc2seg.postprocess import convert_transcripts_to_anndata
from nuc2seg.xenium import (
    load_and_filter_transcripts_as_points,
)
from nuc2seg.utils import get_tile_bounds
from shapely import box

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
        "--use-connected-components",
        help="Use connected components seed segmentation instead of nuclei labels",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--connected-components-min-size",
        help="Minimum size of connected components to consider",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--use-early-stopping",
        help="Use early stopping in greedy expansion algorithm.",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tile-index",
        help="Tile index to process",
        type=int,
        default=None,
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
    return parser


def main():
    args = get_parser().parse_args()
    with h5py.File(args.dataset, "r") as f:
        base_width = f["labels"].shape[0]
        base_height = f["labels"].shape[1]

    if args.tile_index is None:
        dataset = Nuc2SegDataset.load_h5(args.dataset)
        transcripts = load_and_filter_transcripts_as_points(args.transcripts)
        predictions = ModelPredictions.load_h5(args.predictions)
    else:
        dataset = Nuc2SegDataset.load_h5(
            args.dataset,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tile_overlap=args.overlap_percentage,
            tile_index=args.tile_index,
        )

        tile_bbox = box(
            *get_tile_bounds(
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                tile_overlap=args.overlap_percentage,
                tile_index=args.tile_index,
                base_width=base_width,
                base_height=base_height,
            )
        )

        transcripts = load_and_filter_transcripts_as_points(
            args.transcripts, sample_area=tile_bbox
        )
        predictions = ModelPredictions.load_h5(
            args.predictions,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tile_overlap=args.overlap_percentage,
            tile_index=args.tile_index,
        )

    if transcripts is None:
        logger.warning("No transcripts found, exiting")
        return

    celltyping_chains = [CelltypingResults.load_h5(x) for x in args.celltyping_results]
    celltyping_results, aic_scores, bic_scores, best_k = select_best_celltyping_chain(
        celltyping_chains
    )

    result = greedy_cell_segmentation(
        dataset=dataset,
        predictions=predictions,
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        max_expansion_steps=args.max_steps,
        foreground_threshold=args.foreground_prob_threshold,
        use_labels=(not args.use_connected_components),
        min_component_size=args.connected_components_min_size,
        use_early_stopping=args.use_early_stopping,
    )

    logger.info(f"Saving segmentation to {args.output}")
    result.save_h5(args.output)

    gdf = convert_segmentation_to_shapefile(
        segmentation=result.segmentation,
        dataset=dataset,
        predictions=predictions,
        translate=False,
    )

    if gdf is None:
        logger.warning("No cells found in segmentation, exiting")
        return

    gdf["geometry"] = gdf.translate(*dataset.bbox[:2])

    logger.info("Creating anndata")
    ad = convert_transcripts_to_anndata(
        transcript_gdf=transcripts, segmentation_gdf=gdf
    )

    logger.info("Predicting celltypes")

    if len(ad) == 0:
        logger.warning("No cells found in segmentation, skipping celltyping")
        return

    celltype_predictions = predict_celltypes_for_anndata(
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        ad=ad,
        gene_names=celltyping_results.gene_names,
    )

    cell_type_labels = np.argmax(celltype_predictions, axis=1)
    cell_type_labels = pandas.Categorical(
        cell_type_labels,
        categories=sorted(np.unique(cell_type_labels)),
        ordered=True,
    )

    columns = [f"celltype_{i}_prob" for i in range(celltype_predictions.shape[1])]
    celltype_df = pandas.DataFrame(celltype_predictions, columns=columns)
    celltype_df["celltype_assignment"] = cell_type_labels
    celltype_df["segment_id"] = ad.obs["segment_id"].values

    gdf = gdf.merge(
        celltype_df, left_on="segment_id", right_on="segment_id", how="left"
    )
    ad.obs = ad.obs.merge(
        celltype_df, left_on="segment_id", right_on="segment_id", how="left"
    )

    logger.info(f"Saving anndata to {args.anndata_output}")
    ad.write_h5ad(args.anndata_output)

    logger.info(f"Saving shapefile to {args.shapefile_output}")

    gdf.to_parquet(args.shapefile_output)
