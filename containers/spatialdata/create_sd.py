#!/usr/bin/env python
import argparse
import spatialdata_io
import geopandas
import anndata

from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity
from spatialdata.transformations.operations import get_transformation


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
    ad = anndata.read_h5ad(args.anndata)

    shape_transform = get_transformation(sd.shapes["nucleus_boundaries"])

    # Load the segmentation shapefile
    gdf = geopandas.read_parquet(args.segmentation).set_geometry("geometry").dropna()
    gdf_parsed = ShapesModel.parse(gdf, transformations={"global": shape_transform})

    for colname in gdf.columns:
        if colname not in gdf_parsed.columns:
            gdf_parsed[colname] = gdf[colname]

    sd.shapes["nuc2seg_cell_segmentation"] = gdf_parsed

    sd.tables["nuc2seg"] = ad

    sd.write(args.output)


if __name__ == "__main__":
    main()
