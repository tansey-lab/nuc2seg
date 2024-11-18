import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely
import zarr
import pyarrow.parquet
from matplotlib import pyplot as plt
from shapely import Polygon
from skimage.transform import resize
from nuc2seg.utils import (
    drop_invalid_geometries,
    filter_gdf_to_intersects_polygon,
    filter_gdf_to_inside_polygon,
)

logger = logging.getLogger(__name__)


def read_boundaries_into_polygons(
    boundaries_file,
    sample_area: Optional[shapely.Polygon] = None,
    cell_id_column="cell_id",
    x_column_name="vertex_x",
    y_column_name="vertex_y",
    tolerance=100.0,
):
    if sample_area is not None:
        filters = [
            (x_column_name, ">", sample_area.bounds[0] - tolerance),
            (x_column_name, ">", sample_area.bounds[1] - tolerance),
            (y_column_name, "<=", sample_area.bounds[2] + tolerance),
            (y_column_name, "<=", sample_area.bounds[3] + tolerance),
        ]
    else:
        filters = None

    columns = [
        x_column_name,
        y_column_name,
        cell_id_column,
    ]

    boundaries = pyarrow.parquet.read_table(
        boundaries_file, columns=columns, filters=filters, use_threads=True
    )
    boundaries = boundaries.to_pandas()

    geo_df = gpd.GeoDataFrame(
        boundaries,
        geometry=gpd.points_from_xy(
            boundaries[x_column_name], boundaries[y_column_name]
        ),
    )
    polys = geo_df.groupby(cell_id_column)["geometry"].apply(
        lambda x: Polygon(x.tolist())
    )

    gdf = drop_invalid_geometries(gpd.GeoDataFrame(polys))
    gdf = filter_gdf_to_inside_polygon(gdf, sample_area)

    return gdf


def load_transcripts(
    transcripts_file: str,
    sample_area: Optional[shapely.Polygon] = None,
    x_column_name="x_location",
    y_column_name="y_location",
    feature_name_column="feature_name",
    qv_column="qv",
    cell_id_column="cell_id",
    overlaps_nucleus_column="overlaps_nucleus",
):
    if sample_area is not None:
        filters = [
            (x_column_name, ">", sample_area.bounds[0]),
            (x_column_name, ">", sample_area.bounds[1]),
            (y_column_name, "<=", sample_area.bounds[2]),
            (y_column_name, "<=", sample_area.bounds[3]),
        ]
    else:
        filters = None

    columns = (
        [
            feature_name_column,
            x_column_name,
            y_column_name,
            qv_column,
            cell_id_column,
            overlaps_nucleus_column,
        ],
    )

    transcripts_table = pyarrow.parquet.read_table(
        transcripts_file, columns=columns, filters=filters, use_threads=True
    )
    return transcripts_table.to_pandas()


def load_transcripts_as_points(
    transcripts_file: str,
    sample_area: Optional[shapely.Polygon] = None,
    x_column_name="x_location",
    y_column_name="y_location",
    feature_name_column="feature_name",
    qv_column="qv",
    cell_id_column="cell_id",
    overlaps_nucleus_column="overlaps_nucleus",
):
    transcripts = load_transcripts(
        transcripts_file,
        sample_area=sample_area,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        feature_name_column=feature_name_column,
        qv_column=qv_column,
        cell_id_column=cell_id_column,
        overlaps_nucleus_column=overlaps_nucleus_column,
    )

    transcripts[feature_name_column] = transcripts[feature_name_column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    tx_geo_df = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(
            transcripts[x_column_name], transcripts[y_column_name]
        ),
    )

    return tx_geo_df


def load_and_filter_transcripts_as_table(
    transcripts_file: str,
    sample_area: Optional[shapely.Polygon] = None,
    min_qv=20.0,
    x_column_name="x_location",
    y_column_name="y_location",
    feature_name_column="feature_name",
    qv_column="qv",
    cell_id_column="cell_id",
    overlaps_nucleus_column="overlaps_nucleus",
):
    transcripts_df = load_transcripts(
        transcripts_file,
        sample_area=sample_area,
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        feature_name_column=feature_name_column,
        qv_column=qv_column,
        cell_id_column=cell_id_column,
        overlaps_nucleus_column=overlaps_nucleus_column,
    )

    transcripts_df = filter_and_preprocess_transcripts(transcripts_df, min_qv=min_qv)

    return transcripts_df


def load_nuclei(nuclei_file: str, sample_area: Optional[shapely.Polygon] = None):
    nuclei_geo_df = read_boundaries_into_polygons(nuclei_file, sample_area)

    original_n_nuclei = nuclei_geo_df.shape[0]

    nuclei_geo_df = filter_gdf_to_intersects_polygon(nuclei_geo_df, sample_area)

    logger.info(
        f"{original_n_nuclei - nuclei_geo_df.shape[0]} nuclei filtered after bounding to {sample_area}"
    )

    if nuclei_geo_df.empty:
        raise ValueError("No nuclei found in the sample area")

    nuclei_geo_df["nucleus_label"] = np.arange(1, nuclei_geo_df.shape[0] + 1)
    nuclei_geo_df["nucleus_centroid"] = nuclei_geo_df["geometry"].centroid
    nuclei_geo_df["nucleus_centroid_x"] = nuclei_geo_df["geometry"].centroid.x
    nuclei_geo_df["nucleus_centroid_y"] = nuclei_geo_df["geometry"].centroid.y

    logger.info(f"Loaded {nuclei_geo_df.shape[0]} nuclei.")

    return nuclei_geo_df


def filter_and_preprocess_transcripts(transcripts_df, min_qv):
    if np.issubdtype(transcripts_df["cell_id"].dtype, np.integer):
        pass
    else:
        # map string values to integers
        mapping = dict(
            zip(
                transcripts_df["cell_id"].unique(),
                np.arange(1, len(transcripts_df["cell_id"].unique()) + 1),
            )
        )
        if "UNASSIGNED" in mapping:
            del mapping["UNASSIGNED"]
        transcripts_df["cell_id"] = (
            transcripts_df["cell_id"].apply(lambda x: mapping.get(x, 0)).astype(int)
        )

    all_feature_names = transcripts_df["feature_name"].unique()

    to_include_features = set()

    for feature_name in all_feature_names:
        if feature_name.lower().startswith("negcontrolprobe_"):
            continue
        if feature_name.lower().startswith("antisense_"):
            continue
        if feature_name.lower().startswith("negcontrolcodeword_"):
            continue
        if feature_name.lower().startswith("blank_"):
            continue
        if feature_name.startswith("deprecatedcodeword_"):
            continue
        if feature_name.startswith("unassignedcodeword_"):
            continue
        to_include_features.add(feature_name)

    # Filter out controls and low quality transcripts
    transcripts_df = transcripts_df[
        transcripts_df["feature_name"].isin(to_include_features)
    ].copy()
    transcripts_df = transcripts_df[(transcripts_df["qv"] >= min_qv)].copy()

    # Assign a unique integer ID to each gene
    gene_ids = transcripts_df["feature_name"].unique()
    mapping = dict(zip(sorted(gene_ids), np.arange(len(gene_ids))))
    transcripts_df["gene_id"] = transcripts_df["feature_name"].apply(
        lambda x: mapping[x]
    )

    return transcripts_df


def load_and_filter_transcripts_as_points(
    transcripts_file: str, sample_area: Optional[shapely.Polygon] = None, min_qv=20.0
):
    transcripts_df = load_transcripts_as_points(transcripts_file, sample_area)

    transcripts_df = filter_and_preprocess_transcripts(transcripts_df, min_qv=min_qv)

    return transcripts_df


def plot_distribution_of_cell_types(cell_type_probs):
    n_rows = cell_type_probs.shape[1] // 3 + int(cell_type_probs.shape[1] % 3 > 0)
    n_cols = 3
    fig, axarr = plt.subplots(n_rows, n_cols, sharex=True)
    for idx in range(cell_type_probs.shape[1]):
        i, j = idx // 3, idx % 3
        axarr[i, j].hist(
            cell_type_probs[np.argmax(cell_type_probs, axis=1) == idx, idx], bins=200
        )
    plt.show()


def read_xenium_cell_segmentation_masks(
    cell_segment_zarr_file, x_extent_pixels, y_extent_pixels
):
    cells = zarr.open(cell_segment_zarr_file)
    cell_masks = cells.masks[1][:].T.astype(int)
    return resize(
        cell_masks, (x_extent_pixels, y_extent_pixels), order=0, preserve_range=True
    ).astype(int)
