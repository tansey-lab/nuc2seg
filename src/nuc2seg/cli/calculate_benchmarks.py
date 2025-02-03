import argparse
import logging
import pandas
import geopandas
import tqdm
import shapely
import math

from nuc2seg.postprocess import (
    calculate_benchmarks_with_nuclear_prior,
)
from nuc2seg.xenium import load_vertex_file, load_and_filter_transcripts_as_points
from blended_tiling import TilingModule
from shapely.geometry import box

from nuc2seg import log_config
from nuc2seg.utils import generate_tiles

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compare multiple segmentation results."
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--output-file",
        help="Output parquet file to save the results.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segmentation-files",
        help="Segmentation files in GeoParquet format from each method.",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--segmentation-method-names",
        help="Segmentation method names "
        "(in the same order as the entries in `--segmentation-files`).",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--true-boundaries",
        help="Ground truth segmentation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--nuclei-boundaries",
        help="Nuclei segmentations.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tile-size",
        help="Height of the tiles.",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--transcripts",
        help="Transcripts in parquet.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--xenium-cell-metadata",
        help="Cell metadata.",
        type=str,
        required=True,
    )
    return parser


def chunk_shapes(segmentation_gdf, chunk_size):
    if "segment_id" not in segmentation_gdf.columns:
        segmentation_gdf["segment_id"] = range(len(segmentation_gdf))

    tiler = TilingModule(
        tile_size=(chunk_size, chunk_size),
        tile_overlap=(0.0, 0.0),
        base_size=(
            math.ceil(segmentation_gdf.total_bounds[2]),
            math.ceil(segmentation_gdf.total_bounds[3]),
        ),
    )

    bboxes = generate_tiles(
        tiler,
        x_extent=segmentation_gdf.total_bounds[2],
        y_extent=segmentation_gdf.total_bounds[3],
        tile_size=(chunk_size, chunk_size),
        overlap_fraction=0.0,
    )
    centroids = []
    bbox_dict = {}
    for idx, bbox in enumerate(bboxes):
        centroids.append(
            {
                "tile_idx": idx,
                "geometry": shapely.Point(
                    ((bbox[0] + bbox[2]) / 2),
                    ((bbox[1] + bbox[3]) / 2),
                ),
            }
        )
        bbox_dict[idx] = bbox
    logger.info(f"Loaded {len(centroids)} tile centroids")

    centroid_gdf = geopandas.GeoDataFrame(centroids, geometry="geometry")
    joined_to_centroids = geopandas.sjoin_nearest(
        segmentation_gdf,
        centroid_gdf,
    )
    joined_to_centroids = joined_to_centroids.drop_duplicates(
        subset=["segment_id"], keep="first"
    )
    return (
        joined_to_centroids[segmentation_gdf.columns.tolist() + ["tile_idx"]],
        bbox_dict,
    )


def main():
    args = get_parser().parse_args()

    cell_metadata_df = pandas.read_parquet(args.xenium_cell_metadata)
    cell_metadata_gdf = geopandas.GeoDataFrame(
        cell_metadata_df,
        geometry=geopandas.points_from_xy(
            cell_metadata_df["x_centroid"], cell_metadata_df["y_centroid"]
        ),
    )

    dfs = []
    for method_name, seg_fn in zip(
        args.segmentation_method_names, args.segmentation_files
    ):
        logger.info(f"Loading segmentation from {seg_fn}.")
        method_gdf = geopandas.read_parquet(seg_fn)
        method_gdf_chunked, tile_bbox_dict = chunk_shapes(method_gdf, args.tile_size)

        for tile_id, tile_bbox in tqdm.tqdm(
            tile_bbox_dict.items(), total=len(tile_bbox_dict)
        ):
            chunk = method_gdf_chunked[method_gdf_chunked["tile_idx"] == tile_id]

            if chunk.empty:
                continue
            try:
                true_gdf = load_vertex_file(
                    args.true_boundaries, sample_area=box(*tile_bbox)
                )
            except ValueError:
                logger.exception(
                    f"Skipping tile {tile_id}, no ground truth cells in sample area"
                )
                continue

            transcripts = load_and_filter_transcripts_as_points(
                args.transcripts, sample_area=box(*tile_bbox)
            )

            try:
                nuclear_gdf = load_vertex_file(
                    args.nuclei_boundaries, sample_area=box(*tile_bbox)
                )
            except ValueError:
                logger.exception(f"Skipping tile {tile_id}, no nuclei in sample area")
                continue

            true_gdf = geopandas.sjoin(true_gdf, cell_metadata_gdf, how="inner")

            dfs.append(
                calculate_benchmarks_with_nuclear_prior(
                    true_segs=true_gdf,
                    method_segs=chunk,
                    nuclear_segs=nuclear_gdf,
                    transcripts_gdf=transcripts,
                )
            )

    df = pandas.concat(dfs)

    del df["geometry_truth"]
    del df["geometry_nuclear"]
    del df["geometry_method"]
    del df["jaccard_method_segment"]
    del df["jaccard_truth_segment"]

    df.to_parquet(args.output_file)


if __name__ == "__main__":
    main()
