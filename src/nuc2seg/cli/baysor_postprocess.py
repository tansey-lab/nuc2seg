import argparse
import logging
import math
import os.path

import numpy as np
import pandas
import tqdm
from geopandas import GeoDataFrame

from nuc2seg import log_config
from nuc2seg.celltyping import (
    predict_celltypes_for_anndata,
    select_best_celltyping_chain,
)
from nuc2seg.data import CelltypingResults
from nuc2seg.plotting import (
    plot_final_segmentation,
)
from nuc2seg.postprocess import (
    stitch_shapes,
    filter_baysor_shapes_to_most_significant_nucleus_overlap,
    read_baysor_shapefile,
    convert_transcripts_to_anndata,
)
from nuc2seg.utils import get_tile_idx, create_shapely_rectangle
from nuc2seg.xenium import (
    load_and_filter_transcripts_as_points,
    load_vertex_file,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Post process tiled baysor results.")
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--baysor-shapefiles",
        help="One or more shapefiles output by baysor.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--nuclei-file",
        help="Path to the Xenium nuclei boundaries parquet file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--transcripts",
        help="Xenium transcripts in parquet format.",
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
        "--output",
        required=True,
        type=str,
        help="Output path.",
    )
    parser.add_argument(
        "--sample-area",
        default=None,
        type=str,
        help='Crop the dataset to this rectangle, provided in in "x1,y1,x2,y2" format.',
    )
    parser.add_argument(
        "--tile-height",
        help="Height of the tiles.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--tile-width",
        help="Width of the tiles.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--overlap-percentage",
        help="What percent of each tile dimension overlaps with the next tile.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--nucleus-overlap-threshold",
        help="Min amount of area overlapping nucleus for keeping the baysor segment.",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--min-molecules-per-cell",
        help="Dont output cells with less than this many gene counts.",
        type=int,
        default=None,
    )

    return parser


def get_args():
    parser = get_parser()

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    log_config.configure_logging(args)

    transcript_df = load_and_filter_transcripts_as_points(args.transcripts)

    if args.sample_area:
        sample_area = create_shapely_rectangle(
            *[float(x) for x in args.sample_area.split(",")]
        )
    else:
        x_extent = math.ceil(transcript_df["x_location"].astype(float).max())
        y_extent = math.ceil(transcript_df["y_location"].astype(float).max())
        sample_area = create_shapely_rectangle(0, 0, x_extent, y_extent)

    logger.info("Reading baysor results")
    shape_gdfs: list[tuple[int, GeoDataFrame]] = []

    for shapefile_fn in tqdm.tqdm(args.baysor_shapefiles):
        shape_gdf = read_baysor_shapefile(shapes_fn=shapefile_fn)
        tile_idx = get_tile_idx(shapefile_fn)
        shape_gdfs.append((tile_idx, shape_gdf))

    logger.info("Done loading baysor results.")

    logger.info("Stitching shapes")
    stitched_shapes = stitch_shapes(
        shapes=shape_gdfs,
        tile_size=(args.tile_width, args.tile_height),
        sample_area=sample_area,
        overlap=args.overlap_percentage,
    )

    logger.info("Loading nuclei shapes")
    nuclei_geo_df = load_vertex_file(
        fn=args.nuclei_file,
        sample_area=sample_area,
    )

    stitched_shapes.to_parquet(args.output)

    logger.info("Filtering baysor shapes to most significant nucleus overlap")
    baysor_nucleus_intersection = (
        filter_baysor_shapes_to_most_significant_nucleus_overlap(
            baysor_shapes=stitched_shapes,
            nuclei_shapes=nuclei_geo_df,
            overlap_area_threshold=args.nucleus_overlap_threshold,
        )
    )

    logger.info("Saving baysor_nucleus_intersecting_shapes")
    baysor_nucleus_intersection.to_parquet(
        os.path.join(
            os.path.dirname(args.output), "baysor_nucleus_intersecting_shapes.parquet"
        )
    )

    logger.info("Creating anndata")
    adata = convert_transcripts_to_anndata(
        transcript_gdf=transcript_df,
        segmentation_gdf=baysor_nucleus_intersection,
        min_molecules_per_cell=args.min_molecules_per_cell,
    )

    logger.info("Predicting celltypes")
    celltyping_chains = [CelltypingResults.load_h5(x) for x in args.celltyping_results]
    celltyping_results, aic_scores, bic_scores, best_k = select_best_celltyping_chain(
        celltyping_chains
    )
    celltype_predictions = predict_celltypes_for_anndata(
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        ad=adata,
        gene_names=celltyping_results.gene_names,
    )
    cell_type_labels = np.argmax(celltype_predictions, axis=1)
    cell_type_labels = pandas.Categorical(
        cell_type_labels,
        categories=sorted(np.unique(cell_type_labels)),
        ordered=True,
    )

    adata.obs["celltype_assignment"] = cell_type_labels
    baysor_nucleus_intersection = baysor_nucleus_intersection.merge(
        adata.obs[["segment_id", "celltype_assignment"]],
        left_on="segment_id",
        right_on="segment_id",
    )

    for i in range(celltype_predictions.shape[1]):
        adata.obs[f"celltype_{i}_prob"] = celltype_predictions[:, i]
        baysor_nucleus_intersection = baysor_nucleus_intersection.merge(
            adata.obs[["segment_id", f"celltype_{i}_prob"]],
            left_on="segment_id",
            right_on="segment_id",
        )

    logger.info("Saving anndata")
    adata.write_h5ad(os.path.join(os.path.dirname(args.output), "anndata.h5ad"))

    logger.info("Plotting final segmentation")
    plot_final_segmentation(
        nuclei_gdf=nuclei_geo_df,
        segmentation_gdf=baysor_nucleus_intersection,
        output_path=os.path.join(os.path.dirname(args.output), "segmentation.png"),
    )
