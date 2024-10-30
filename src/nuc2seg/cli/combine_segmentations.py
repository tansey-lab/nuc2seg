import argparse
import logging
import os.path

import anndata
import geopandas as gpd
import numpy as np
import pandas
import shapely
from blended_tiling import TilingModule

from nuc2seg import log_config
from nuc2seg.data import Nuc2SegDataset, SegmentationResults
from nuc2seg.data import generate_tiles
from nuc2seg.plotting import (
    plot_segmentation_class_assignment,
    celltype_histogram,
    celltype_area_violin,
)
from nuc2seg.segment import segmentation_array_to_shapefile
from nuc2seg.utils import get_tile_idx

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Combine tiled segmentation results into single output files."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-dir",
        help="Directory to save combined files.",
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
        "--segmentation-outputs",
        help="Cell segmentation outputs in h5 format.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--adatas",
        help="adata outputs in h5ad format.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--shapes",
        help="shapefile outputs in parquet format.",
        type=str,
        required=True,
        nargs="+",
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


def combine_segmentation_results(
    h5_fns: list[str],
    anndata_fns: list[str],
    shapefile_fns: list[str],
    labels: np.ndarray,
    tile_size: tuple[int, int],
    overlap: float,
    base_size: tuple[int, int],
):
    h5_fns = sorted(h5_fns)
    anndata_fns = sorted(anndata_fns)
    shapefile_fns = sorted(shapefile_fns)

    if len(h5_fns) != len(anndata_fns) != len(shapefile_fns):
        raise ValueError("Number of h5 files, shapefiles, anndata files must match")
    fns = zip(h5_fns, anndata_fns, shapefile_fns)
    stitched_result = labels.copy()
    stitched_result[stitched_result > 0] = -1

    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=base_size,
    )

    bboxes = generate_tiles(
        tiler,
        x_extent=base_size[0],
        y_extent=base_size[1],
        tile_size=tile_size,
        overlap_fraction=overlap,
    )

    centroids = []

    bbox_dict = {}

    for idx, bbox in enumerate(bboxes):
        centroids.append(
            {
                "tile_idx": idx,
                "geometry": shapely.Point(
                    (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                ),
            }
        )
        bbox_dict[idx] = bbox
    logger.info(f"Loaded {len(centroids)} tile centroids")

    centroid_gdf = gpd.GeoDataFrame(centroids, geometry="geometry")

    current_n_segments = 0

    concatenated_anndata = None

    gpd_results = []

    for h5_fn, anndata_fn, shapefile_fn in fns:
        tile_idx = get_tile_idx(h5_fn)

        segment_gpd = gpd.read_parquet(shapefile_fn)

        segmentation_result = SegmentationResults.load_h5(h5_fn).segmentation

        segmentation_gdf = segmentation_array_to_shapefile(segmentation_result)
        joined_to_centroids = gpd.sjoin_nearest(
            segmentation_gdf,
            centroid_gdf,
        )
        joined_to_centroids = joined_to_centroids.drop_duplicates(subset=["segment_id"])
        n_segments = (joined_to_centroids["tile_idx"] == tile_idx).sum()
        segments_to_filter = joined_to_centroids[
            joined_to_centroids["tile_idx"] != tile_idx
        ].segment_id.unique()

        ad = anndata.read_h5ad(anndata_fn)
        ad = ad[~ad.obs["segment_id"].isin(segments_to_filter)].copy()

        new_segment_id_map = dict(
            zip(
                sorted(ad.obs["segment_id"].unique()),
                range(current_n_segments, current_n_segments + n_segments),
            )
        )
        ad.obs["segment_id"] = ad.obs["segment_id"].map(new_segment_id_map)

        segment_gpd = segment_gpd[
            ~segment_gpd["segment_id"].isin(segments_to_filter)
        ].copy()
        segment_gpd["segment_id"] = segment_gpd["segment_id"].map(new_segment_id_map)
        gpd_results.append(segment_gpd)

        if concatenated_anndata:
            concatenated_anndata = anndata.concat([concatenated_anndata, ad])
        else:
            concatenated_anndata = ad

        current_n_segments += n_segments

    return (
        concatenated_anndata,
        pandas.concat(gpd_results),
    )


def main():
    args = get_parser().parse_args()

    dataset: Nuc2SegDataset = Nuc2SegDataset.load_h5(args.dataset)

    concatenated_anndata, gdf = combine_segmentation_results(
        h5_fns=args.segmentation_outputs,
        anndata_fns=args.adatas,
        shapefile_fns=args.shapes,
        labels=dataset.labels,
        tile_size=(args.tile_height, args.tile_width),
        overlap=args.overlap_percentage,
        base_size=dataset.labels.shape,
    )

    concatenated_anndata.write_h5ad(os.path.join(args.output_dir, "anndata.h5ad"))

    gdf.to_parquet(os.path.join(args.output_dir, "shapes.parquet"))

    logger.info(f"Plotting segmentation and class assignment.")
    plot_segmentation_class_assignment(
        segmentation_gdf=gdf,
        output_path=os.path.join(
            os.path.dirname(args.output_dir), "class_assignment.png"
        ),
        cat_column="celltype_assignment",
    )
    celltype_area_violin(
        segmentation_gdf=gdf,
        output_path=os.path.join(
            os.path.dirname(args.output_dir), "celltype_area_violin.pdf"
        ),
        cat_column="celltype_assignment",
    )
    celltype_histogram(
        segmentation_gdf=gdf,
        output_path=os.path.join(
            os.path.dirname(args.output_dir), "celltype_histograms.pdf"
        ),
        cat_column="celltype_assignment",
    )
