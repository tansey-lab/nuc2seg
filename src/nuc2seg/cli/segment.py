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
    ray_tracing_cell_segmentation,
)
from nuc2seg.postprocess import convert_transcripts_to_anndata
from nuc2seg.xenium import (
    load_and_filter_transcripts_as_points,
)
from nuc2seg.utils import get_tile_bounds
from nuc2seg.utils import create_shapely_rectangle
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
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--n-celltypes",
        default=None,
        type=int,
        help="Force using this number of celltypes, otherwise pick via BIC.",
    )
    parser.add_argument(
        "--expansion-method",
        default="greedy",
        choices=("greedy", "ray_tracing"),
        help="Choose method for expansion.",
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

    with h5py.File(args.dataset, "r") as f:
        base_width = f["labels"].shape[0]
        base_height = f["labels"].shape[1]

    if args.tile_index is None:
        dataset = Nuc2SegDataset.load_h5(args.dataset)
        transcripts = load_and_filter_transcripts_as_points(args.transcripts)
        predictions = ModelPredictions.load_h5(args.predictions)
        if sample_area:
            slide_bbox = sample_area.bounds
        else:
            slide_bbox = dataset.bbox
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

        if sample_area:
            slide_bbox = (
                (tile_bbox.bounds[0] * dataset.resolution) + sample_area.bounds[0],
                (tile_bbox.bounds[1] * dataset.resolution) + sample_area.bounds[1],
                (tile_bbox.bounds[2] * dataset.resolution) + sample_area.bounds[0],
                (tile_bbox.bounds[3] * dataset.resolution) + sample_area.bounds[1],
            )
        else:
            slide_bbox = tile_bbox.bounds

        transcripts = load_and_filter_transcripts_as_points(
            args.transcripts, sample_area=box(*slide_bbox)
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
    if args.n_celltypes:
        best_k = list(celltyping_chains[0].n_component_values).index(args.n_celltypes)
        celltyping_results, aic_scores, bic_scores, best_k = (
            select_best_celltyping_chain(celltyping_chains, best_k)
        )
    else:
        celltyping_results, aic_scores, bic_scores, best_k = (
            select_best_celltyping_chain(celltyping_chains, None)
        )

    if args.expansion_method == "greedy":
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
    elif args.expansion_method == "ray_tracing":
        result = ray_tracing_cell_segmentation(
            dataset=dataset,
            predictions=predictions,
            prior_probs=celltyping_results.prior_probs[best_k],
            expression_profiles=celltyping_results.expression_profiles[best_k],
            max_length=args.max_steps,
            foreground_threshold=args.foreground_prob_threshold,
            use_labels=(not args.use_connected_components),
            use_early_stopping=args.use_early_stopping,
        )
    else:
        raise ValueError(args.expansion_method)

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

    gdf["geometry"] = gdf.translate(*slide_bbox[:2])
    gdf["geometry"] = gdf.geometry.scale(
        xfact=dataset.resolution,
        yfact=dataset.resolution,
        origin=(slide_bbox[0], slide_bbox[1]),
    )

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
