#!/usr/bin/env python
import argparse
import spatialdata_io
import geopandas


def get_args():
    parser = argparse.ArgumentParser(description="Create a spatialdata archive.")
    parser.add_argument(
        "--segmentation", type=str, help="The segmentation shapefile file."
    )
    parser.add_argument("--xenium-dir", type=str, help="Path to the xenium project")
    parser.add_argument("--output", type=str, help="The output zarr file")
    return parser.parse_args()


def main():
    args = get_args()
    sd = spatialdata_io.xenium(args.xenium_dir)

    # Load the segmentation shapefile
    seg = geopandas.read_parquet(args.segmentation)

    sd.shapes["nuc2seg_cell_segmentation"] = seg

    sd.write(args.output)


if __name__ == "__main__":
    main()
