import argparse
import logging
import anndata
import numpy as np
import pandas
import geopandas
import shapely

from nuc2seg import log_config
from nuc2seg.celltyping import (
    select_best_celltyping_chain,
    predict_celltypes_for_anndata_with_noise_type,
)
from nuc2seg.data import CelltypingResults, Nuc2SegDataset
from nuc2seg.preprocessing import create_rasterized_dataset
from nuc2seg.xenium import (
    load_vertex_file,
    load_and_filter_transcripts_as_points,
)
from nuc2seg.segment import segmentation_array_to_shapefile
from nuc2seg.constants import NOISE_CELLTYPE
from nuc2seg.utils import (
    create_shapely_rectangle,
    transform_shapefile_to_slide_space,
    get_roi,
)
from nuc2seg.plotting import plot_preprocessing

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
        "--adata",
        type=str,
        help="Anndata containing prior segmented data.",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output path.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--labels-output",
        help="Training labels output as a GeoParquet shapefile. This shows which pixels were labeled for training",
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
        default=0,
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
        help="Number of transcripts in a pixel to be considered foreground.",
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
        type=int,
        help="Force using this number of celltypes, otherwise pick via BIC.",
    )
    parser.add_argument(
        "--min-celltyping-transcripts",
        help="Threshold for determining whether a cell can be labeled with a celltype, "
        "cells with less than this many transcripts will be more likely to remain unlabeled.",
        type=int,
        default=10,
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
    ).reset_index(drop=True)
    if "segment_id" in nuclei_geo_df.columns:
        del nuclei_geo_df["segment_id"]
    nuclei_geo_df.reset_index(names="segment_id", inplace=True)
    nuclei_geo_df["segment_id"] += 1

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

    n_celltypes = celltyping_results.n_component_values[best_k]
    logger.info("Predicting celltypes for segments and transcripts")
    adata = anndata.read_h5ad(args.adata)
    celltype_predictions = predict_celltypes_for_anndata_with_noise_type(
        prior_probs=celltyping_results.prior_probs[best_k],
        expression_profiles=celltyping_results.expression_profiles[best_k],
        ad=adata,
        min_transcripts=args.min_celltyping_transcripts,
    )
    cell_type_labels = np.argmax(celltype_predictions, axis=1)
    nucleus_centroids = adata.obsm["spatial"]
    points = [shapely.geometry.Point(x, y) for x, y in nucleus_centroids]
    nucleus_celltype_geodf = geopandas.GeoDataFrame(geometry=points)
    nucleus_celltype_geodf["celltype"] = cell_type_labels

    rasterized_dataset = create_rasterized_dataset(
        prior_segmentation_gdf=nuclei_geo_df,
        tx_geo_df=tx_geo_df,
        sample_area=sample_area,
        resolution=args.resolution,
        foreground_distance=args.foreground_nucleus_distance,
        background_distance=args.background_nucleus_distance,
        background_pixel_transcripts=args.background_pixel_transcripts,
        background_transcript_distance=args.background_transcript_distance,
    )

    celltypes = (
        geopandas.sjoin_nearest(nuclei_geo_df, nucleus_celltype_geodf)
        .drop_duplicates(subset="segment_id")
        .set_index("segment_id")["celltype"]
    )
    n_noisy_cells = (celltypes == NOISE_CELLTYPE).sum()
    logger.info(f"Found {n_noisy_cells} noisy cells, will not label")
    segment_id_to_celltype = celltypes.to_dict()

    # Assign hard labels to nuclei
    class_labels = np.zeros_like(rasterized_dataset.labels, dtype=int)
    for segment_id in np.unique(rasterized_dataset.labels):
        if segment_id < 1:
            continue
        celltype_for_segment = segment_id_to_celltype.get(segment_id, 0)

        class_labels[rasterized_dataset.labels == segment_id] = celltype_for_segment
    ds = Nuc2SegDataset(
        labels=rasterized_dataset.labels,
        angles=rasterized_dataset.angles,
        classes=class_labels,
        transcripts=rasterized_dataset.transcripts,
        bbox=rasterized_dataset.bbox,
        n_classes=n_celltypes,
        n_genes=rasterized_dataset.n_genes,
        resolution=rasterized_dataset.resolution,
    )

    logger.info("Saving to h5")
    ds.save_h5(args.output)

    logger.info("Plotting preprocessing")
    roi_bbox = get_roi(ds.resolution, ds.labels)
    plot_preprocessing(
        dataset=ds.clip(roi_bbox),
        output_path=args.output.replace(".h5", "_preprocessing.pdf"),
    )

    logger.info("Saving labels to GeoParquet")
    labels_shapefile = segmentation_array_to_shapefile(rasterized_dataset.labels)
    labels_shapefile = transform_shapefile_to_slide_space(
        labels_shapefile, ds.resolution, ds.bbox
    )
    labels_shapefile.to_parquet(args.labels_output)
