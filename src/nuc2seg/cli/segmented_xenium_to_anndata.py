from nuc2seg.postprocess import convert_transcripts_to_anndata
from nuc2seg.xenium import load_and_filter_transcripts_as_points, load_vertex_file
from nuc2seg import log_config

import argparse
import tqdm
import logging
import anndata


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="This is a utility for creating an anndata file from nuclear or cell segmented Xenium data"
    )
    log_config.add_logging_args(parser)
    parser.add_argument(
        "--transcripts-file",
        help="Path to the Xenium transcripts parquet file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vertex-file",
        help="Path to the Xenium boundaries parquet file",
        type=str,
        required=True,
    )
    parser.add_argument("--chunk-size", help="Chunk size", type=int, default=100_000)
    parser.add_argument(
        "--output",
        help="Output .h5ad file",
        type=str,
        required=True,
    )
    return parser


def main():
    args = get_args().parse_args()
    transcripts = load_and_filter_transcripts_as_points(args.transcripts_file)
    segments = load_vertex_file(args.vertex_file)

    logger.info(f"Read {len(transcripts)} transcripts and {len(segments)} segments")

    transcript_chunk_size = args.chunk_size

    logger.info(f"Converting transcripts to anndata")

    ads = []
    for i in tqdm.tqdm(range(0, len(transcripts), transcript_chunk_size)):
        ad = convert_transcripts_to_anndata(
            transcript_gdf=transcripts[i : i + transcript_chunk_size],
            segmentation_gdf=segments,
            min_molecules_per_cell=1,
        )
        ads.append(ad)

        for colname in ad.obs.columns:
            if colname.endswith("_centroid"):
                del ad.obs[colname]

    adata = anndata.concat(ads, join="outer")

    adata.write_h5ad(args.output)
