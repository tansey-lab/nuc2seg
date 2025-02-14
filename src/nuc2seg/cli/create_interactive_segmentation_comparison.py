import argparse
import logging

import geopandas
import pandas

from nuc2seg import log_config
from nuc2seg.plotting import create_interactive_segmentation_comparison
from nuc2seg.utils import create_shapely_rectangle
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

    logger.info(f"Loading cell metadata from {args.xenium_cell_metadata}.")
    cell_metadata_df = pandas.read_parquet(args.xenium_cell_metadata)
    cell_metadata_gdf = geopandas.GeoDataFrame(
        cell_metadata_df,
        geometry=geopandas.points_from_xy(
            cell_metadata_df["x_centroid"], cell_metadata_df["y_centroid"]
        ),
    )
    cell_metadata_gdf = cell_metadata_gdf.clip(sample_area)
    logger.info(f"Loading true boundaries from {args.true_boundaries}.")
    true_gdf = load_vertex_file(args.true_boundaries, sample_area=sample_area)
    logger.info(f"Loading transcripts from {args.transcripts}.")

    polygon_layers = []
    names = []

    logger.info("Separating groundtruth segments by segmentation method.")
    true_gdf = geopandas.sjoin(true_gdf, cell_metadata_gdf, how="inner")
    gb = true_gdf.groupby("segmentation_method")
    for group_name in gb.groups:
        df = gb.get_group(group_name)
        del df["nucleus_centroid"]
        polygon_layers.append(df)
        names.append(group_name)

    transcripts = load_and_filter_transcripts_as_points(
        args.transcripts, sample_area=sample_area
    )
    logger.info(f"Loading nuclei boundaries from {args.nuclei_boundaries}.")
    nuclear_gdf = load_vertex_file(args.nuclei_boundaries, sample_area=sample_area)

    polygon_layers.append(nuclear_gdf)
    names.append("nuclei")

    for method_name, seg_fn in zip(
        args.segmentation_method_names, args.segmentation_files
    ):
        logger.info(f"Loading segmentation from {seg_fn}.")
        method_gdf = geopandas.read_parquet(seg_fn)
        method_gdf = method_gdf.clip(sample_area)

        polygon_layers.append(method_gdf)
        names.append(method_name)

    create_interactive_segmentation_comparison(
        polygon_gdfs=polygon_layers,
        names=names,
        points_gdf=transcripts,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()
