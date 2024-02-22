#!/usr/bin/env python
import argparse
import spatialdata_io
import geopandas
import anndata


def get_args():
    parser = argparse.ArgumentParser(description="Create a spatialdata archive.")
    parser.add_argument(
        "--segmentation", type=str, help="The segmentation shapefile file."
    )
    parser.add_argument("--xenium-dir", type=str, help="Path to the xenium project")
    parser.add_argument("--anndata", type=str, help="Path to the anndata file")
    parser.add_argument("--output", type=str, help="The output zarr file")
    return parser.parse_args()


def main():
    args = get_args()
    sd = spatialdata_io.xenium(args.xenium_dir)

    # Load the segmentation shapefile
    seg = geopandas.read_parquet(args.segmentation)

    # Load the anndata file
    ad = anndata.read_h5ad(args.anndata)

    sd.shapes["nuc2seg_cell_segmentation"] = seg
    sd.table = ad

    sd.write(args.output)
