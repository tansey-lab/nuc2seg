import argparse
import logging
import os

import pandas
import geopandas
import tqdm

from nuc2seg import log_config
from nuc2seg.postprocess import (
    calculate_benchmarks_with_nuclear_prior,
)
from nuc2seg.xenium import load_vertex_file, load_and_filter_transcripts_as_points

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
    parser.add_argument("--chunk-size", help="Chunk size", type=int, default=5_000)
    return parser


def main():
    args = get_parser().parse_args()

    logger.info("Loading true boundaries.")
    true_gdf = load_vertex_file(args.true_boundaries)

    logger.info("Loading transcripts.")
    transcripts = load_and_filter_transcripts_as_points(args.transcripts)

    nuclear_gdf = load_vertex_file(args.nuclei_boundaries)

    cell_metadata_df = pandas.read_parquet(args.xenium_cell_metadata)
    cell_metadata_gdf = geopandas.GeoDataFrame(
        cell_metadata_df,
        geometry=geopandas.points_from_xy(
            transcripts["x_centroid"], transcripts["y_centroid"]
        ),
    )

    true_gdf = geopandas.sjoin(true_gdf, cell_metadata_gdf, how="inner")

    os.makedirs(args.output_dir, exist_ok=True)
    dfs = []
    for method_name, seg_fn in zip(
        args.segmentation_method_names, args.segmentation_files
    ):
        logger.info(f"Loading segmentation from {seg_fn}.")
        method_gdf = load_vertex_file(seg_fn)

        segments_chunk_size = args.chunk_size

        logger.info(f"Converting transcripts to anndata")
        logger.info(f"Calculating benchmarks for {method_name}.")

        for i in tqdm.tqdm(range(0, len(method_gdf), segments_chunk_size)):
            chunk = method_gdf[i : i + segments_chunk_size]

            dfs.append(
                calculate_benchmarks_with_nuclear_prior(
                    true_segs=true_gdf,
                    method_segs=chunk,
                    nuclear_segs=nuclear_gdf,
                    transcripts_gdf=transcripts,
                )
            )

    df = pandas.concat(dfs)

    df.to_parquet(args.output_file)
