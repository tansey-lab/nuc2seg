#!/usr/bin/env python
import argparse
import spatialdata
import json
import numpy as np

from shapely.affinity import affine_transform


def transform_geodataframe(gdf, matrix):
    """
    Apply a 3x3 affine transformation matrix to all geometries in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    matrix (numpy.ndarray): 3x3 affine transformation matrix

    Returns:
    GeoDataFrame: Transformed GeoDataFrame
    """
    # Extract the 6 needed values from the 3x3 matrix
    # [a b xoff]    ->  (a, b, d, e, xoff, yoff)
    # [d e yoff]
    # [0 0  1  ]
    transform_params = (
        matrix[0, 0],
        matrix[0, 1],  # a, b
        matrix[1, 0],
        matrix[1, 1],  # d, e
        matrix[0, 2],
        matrix[1, 2],  # xoff, yoff
    )

    transformed_gdf = gdf.copy()
    transformed_gdf["geometry"] = transformed_gdf["geometry"].apply(
        lambda geom: affine_transform(geom, transform_params)
    )
    return transformed_gdf


def get_args():
    parser = argparse.ArgumentParser(description="Create a geoparquet file from sd.")
    parser.add_argument("--zarr", type=str, help="Zarr spatialdata file.")
    parser.add_argument("--xenium-experiment", type=str, help=".experiment file.")
    parser.add_argument(
        "--shapes-key",
        type=str,
        help="Key for the shapes in the SD file.",
        default="cellpose_boundaries",
    )
    parser.add_argument("--output", type=str, help="The output GeoParquet file")
    return parser.parse_args()


def main():
    args = get_args()

    sd = spatialdata.SpatialData.load(args.zarr)
    with open(args.xenium_experiment) as f:
        experiment = json.load(f)
        pixel_size = experiment["pixel_size"]

    gdf = sd.shapes[args.shapes_key]

    transformation_matrix = np.array(
        [[pixel_size, 0.0, 0.0], [0.0, pixel_size, 0.0], [0.0, 0.0, 1.0]]
    )

    transformed_gdf = transform_geodataframe(gdf, transformation_matrix)
    transformed_gdf.to_parquet(args.output)
