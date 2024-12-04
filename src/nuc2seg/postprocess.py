import json
import logging
import math

import anndata
import geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
from blended_tiling import TilingModule
from scipy.sparse import csr_matrix
from shapely import box

from nuc2seg.segment import logger
from nuc2seg.utils import generate_tiles, spatial_join_polygons_and_transcripts

logger = logging.getLogger(__name__)


def filter_gdf_to_tile_boundary(
    gdf: gpd.GeoDataFrame, tile_idx: int, tile_size, base_size, overlap
):
    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=base_size,
    )
    tile_masks = tiler.get_tile_masks()[:, 0, :, :]
    bboxes = generate_tiles(
        tiler,
        x_extent=base_size[0],
        y_extent=base_size[1],
        tile_size=tile_size,
        overlap_fraction=overlap,
    )

    masks_and_bboxes = list(zip(tile_masks, bboxes))

    mask = masks_and_bboxes[tile_idx][0].detach().cpu().numpy()
    bbox = masks_and_bboxes[tile_idx][1]

    mask = (mask > 0.5).astype(bool)
    x, y = np.where(mask)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    offset_x = bbox[0]
    offset_y = bbox[1]

    selection_box = box(
        x_min + offset_x,
        y_min + offset_y,
        x_max + offset_x + 1,
        y_max + offset_y + 1,
    )

    gdf["intersection_area"] = gdf.geometry.apply(
        lambda g: g.intersection(selection_box).area
    )

    # Step 2: Calculate the percentage of the intersection area relative to the polygon's own area
    gdf["intersection_percentage"] = gdf["intersection_area"] / gdf.geometry.area

    return gdf[gdf["intersection_percentage"] > 0.5]


def stitch_shapes(
    shapes: list[tuple[int, gpd.GeoDataFrame]],
    tile_size,
    sample_area: shapely.Polygon,
    overlap,
):
    x_extent = math.ceil(sample_area.bounds[2] - sample_area.bounds[0])
    y_extent = math.ceil(sample_area.bounds[3] - sample_area.bounds[1])

    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=(x_extent, y_extent),
    )

    bboxes = generate_tiles(
        tiler,
        x_extent=x_extent,
        y_extent=y_extent,
        tile_size=tile_size,
        overlap_fraction=overlap,
    )

    centroids = []

    for idx, bbox in enumerate(bboxes):
        centroids.append(
            {
                "tile_idx": idx,
                "geometry": shapely.Point(
                    ((bbox[0] + bbox[2]) / 2) + sample_area.bounds[0],
                    ((bbox[1] + bbox[3]) / 2) + sample_area.bounds[1],
                ),
            }
        )
    logger.info(f"Loaded {len(centroids)} tile centroids")

    centroid_gdf = gpd.GeoDataFrame(centroids, geometry="geometry")

    results = []
    raw_baysor_segments = 0

    for tile_idx, shapefile in shapes:
        raw_baysor_segments += len(shapefile)
        joined_to_centroids = gpd.sjoin_nearest(
            shapefile,
            centroid_gdf,
        )
        # dedupe joined_to_centroids
        joined_to_centroids = joined_to_centroids.drop_duplicates(subset=["cell"])

        filtered_shapes = joined_to_centroids[
            joined_to_centroids["tile_idx"] == tile_idx
        ]

        results.append(filtered_shapes)
    logger.info(f"Loaded {raw_baysor_segments} raw baysor segments")

    result_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

    logger.info(f"After stitching, {len(result_gdf)} segments remain")

    if "index_right" in result_gdf:
        del result_gdf["index_right"]
    if "index_left" in result_gdf:
        del result_gdf["index_left"]

    result_gdf.reset_index(drop=False, names="segment_id", inplace=True)

    return result_gdf


def read_baysor_shapefile(shapes_fn):
    with open(shapes_fn) as f:
        geojson_data = json.load(f)

    records = []
    for geometry in geojson_data["geometries"]:
        if len(geometry["coordinates"][0]) <= 3:
            logger.debug(
                f"Skipping cell with {len(geometry['coordinates'][0])} vertices"
            )
            continue
        polygon = shapely.Polygon(geometry["coordinates"][0])
        records.append({"geometry": polygon, "cell": geometry["cell"]})

    return gpd.GeoDataFrame(records, geometry="geometry")


def read_baysor_shapes_with_cluster_assignment(
    shapes_fn, transcripts_fn, x_column_name="x", y_column_name="y"
) -> gpd.GeoDataFrame:
    with open(shapes_fn) as f:
        geojson_data = json.load(f)

    records = []
    for geometry in tqdm.tqdm(geojson_data["geometries"]):
        if len(geometry["coordinates"][0]) <= 3:
            logger.debug(
                f"Skipping cell with {len(geometry['coordinates'][0])} vertices"
            )
            continue
        polygon = shapely.Polygon(geometry["coordinates"][0])
        records.append({"geometry": polygon, "cell": geometry["cell"]})

    gdf = gpd.GeoDataFrame(records)

    transcripts_df = pd.read_csv(
        transcripts_fn,
        usecols=["cell", "cluster", "gene", "assignment_confidence", "x", "y"],
    )
    tx_geo_df = gpd.GeoDataFrame(
        transcripts_df,
        geometry=gpd.points_from_xy(
            transcripts_df[x_column_name], transcripts_df[y_column_name]
        ),
    )

    transcripts_df["cell_id"] = transcripts_df["cell"].apply(
        lambda x: int(x.split("-")[-1])
    )
    cell_to_cluster = transcripts_df[["cell_id", "cluster"]].drop_duplicates()

    result = gdf.merge(cell_to_cluster, left_on="cell", right_on="cell_id")
    del result["cell_id"]
    return result, tx_geo_df


def filter_baysor_shapes_to_most_significant_nucleus_overlap(
    baysor_shapes,
    nuclei_shapes,
    overlap_area_threshold=2.0,
    id_col="segment_id",
    nucleus_overlap_area_col="nucleus_overlap_area",
):
    overlay_gdf = gpd.overlay(baysor_shapes, nuclei_shapes, how="intersection")
    overlay_gdf[nucleus_overlap_area_col] = overlay_gdf.geometry.area

    overlay_gdf = overlay_gdf[
        overlay_gdf[nucleus_overlap_area_col] > overlap_area_threshold
    ]
    gb = overlay_gdf.groupby(id_col)[[nucleus_overlap_area_col]].max()

    return baysor_shapes.merge(gb, left_on=id_col, right_index=True)


def calculate_segmentation_jaccard_index(
    transcripts: gpd.GeoDataFrame,
    segmentation_a: gpd.GeoDataFrame,
    segmentation_b: gpd.GeoDataFrame,
    overlap_area_col="overlap_area",
    overlap_area_threshold=2.0,
):
    segmentation_a = segmentation_a.reset_index(names="segment_id_a")
    segmentation_b = segmentation_b.reset_index(names="segment_id_b")

    overlay_gdf = gpd.overlay(
        segmentation_a, segmentation_b, how="intersection", keep_geom_type=False
    )
    overlay_gdf[overlap_area_col] = overlay_gdf.geometry.area

    overlay_gdf = overlay_gdf[overlay_gdf[overlap_area_col] > overlap_area_threshold]

    to_select = (
        overlay_gdf.groupby("segment_id_a")[[overlap_area_col]]
        .idxmax()
        .values.squeeze()
    )

    if to_select.shape == ():
        to_select = [to_select]

    max_overlay_gdf = overlay_gdf.loc[to_select, :]

    segment_a_to_segment_b_map = (
        max_overlay_gdf[["segment_id_a", "segment_id_b"]]
        .set_index("segment_id_a")["segment_id_b"]
        .to_dict()
    )

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_a, transcripts=transcripts
    )
    sjoined_gdf.reset_index(inplace=True, drop=False, names="index")

    segment_id_to_transcripts_a = (
        sjoined_gdf[["segment_id_a", "index_right"]]
        .groupby("segment_id_a")
        .agg({"index_right": set})["index_right"]
        .to_dict()
    )

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_b, transcripts=transcripts
    )

    segment_id_to_transcripts_b = (
        sjoined_gdf[["segment_id_b", "index_right"]]
        .groupby("segment_id_b")
        .agg({"index_right": set})["index_right"]
        .to_dict()
    )

    results = []

    for segment_id_a, segment_id_b in segment_a_to_segment_b_map.items():
        segment_a_transcripts = segment_id_to_transcripts_a.get(segment_id_a, set())
        segment_b_transcripts = segment_id_to_transcripts_b.get(segment_id_b, set())

        intersection = len(segment_a_transcripts.intersection(segment_b_transcripts))
        union = len(segment_a_transcripts.union(segment_b_transcripts))

        if union == 0:
            continue

        results.append(
            {
                "segment_id_a": segment_id_a,
                "segment_id_b": segment_id_b,
                "jaccard_index": intersection / union,
            }
        )
    return pd.DataFrame(results)


def calculate_average_intersection_over_union(
    seg_a, seg_b, overlap_area_threshold=2.0, overlap_area_col="intersection"
):
    seg_a = seg_a.reset_index(names="segment_id_a")
    seg_b = seg_b.reset_index(names="segment_id_b")

    overlay_gdf = geopandas.overlay(
        seg_a, seg_b, how="intersection", keep_geom_type=False
    )
    overlay_gdf[overlap_area_col] = overlay_gdf.geometry.area

    overlay_gdf = overlay_gdf[overlay_gdf[overlap_area_col] > overlap_area_threshold]
    max_overlay_gdf = overlay_gdf.loc[
        overlay_gdf.groupby("segment_id_a")[[overlap_area_col]]
        .idxmax()
        .values.squeeze(),
        :,
    ]

    max_overlay_gdf = max_overlay_gdf[
        ["segment_id_a", "segment_id_b", overlap_area_col]
    ]

    result = max_overlay_gdf.merge(
        seg_a[["geometry", "segment_id_a"]],
        left_on="segment_id_a",
        right_on="segment_id_a",
    ).merge(
        seg_b[["geometry", "segment_id_b"]],
        left_on="segment_id_b",
        right_on="segment_id_b",
    )

    result["union"] = result.apply(
        lambda row: row["geometry_x"].union(row["geometry_y"]).area, axis=1
    )

    result["iou"] = result.apply(
        lambda row: row[overlap_area_col] / row["union"], axis=1
    )

    return result


def join_segments_on_max_overlap(
    segs_a,
    segs_b,
    segs_a_id_column=None,
    segs_b_id_column=None,
    geometry_a_column=None,
    geometry_b_column=None,
    overlap_area_column="overlap_area",
):
    if segs_a_id_column is None:
        segs_a_id_column = "segment_id_a"
        segs_a = segs_a.reset_index(names=segs_a_id_column)

    if segs_b_id_column is None:
        segs_b_id_column = "segment_id_b"
        segs_b = segs_b.reset_index(names=segs_b_id_column)

    if geometry_a_column is None:
        geometry_a_column = "geometry_a"

    if geometry_b_column is None:
        geometry_b_column = "geometry_b"

    overlay_gdf = geopandas.overlay(
        segs_a, segs_b, how="intersection", keep_geom_type=False
    )
    overlay_gdf[overlap_area_column] = overlay_gdf.geometry.area

    max_overlay_gdf = overlay_gdf.loc[
        overlay_gdf.groupby("truth_segment_id")[[overlap_area_column]]
        .idxmax()
        .values.squeeze(),
        :,
    ]

    max_overlay_gdf = max_overlay_gdf[
        [segs_a_id_column, segs_b_id_column, overlap_area_column]
    ].drop_duplicates(subset=[segs_a_id_column, segs_b_id_column])

    return (
        max_overlay_gdf.merge(
            segs_a[["geometry", segs_a_id_column]],
            left_on=segs_a_id_column,
            right_on=segs_a_id_column,
        )
        .merge(
            segs_b[["geometry", segs_b_id_column]],
            left_on=segs_b_id_column,
            right_on=segs_b_id_column,
        )
        .rename(
            columns={"geometry_x": geometry_a_column, "geometry_y": geometry_b_column}
        )
    )


def calculate_benchmarks_with_nuclear_prior(
    true_segs, method_segs, nuclear_segs, transcripts_gdf
):
    true_segs = true_segs.reset_index(names="truth_segment_id")
    method_segs = method_segs.reset_index(names="method_segment_id")
    nuclear_segs = nuclear_segs.reset_index(names="nuclear_segment_id")

    truth_to_method = join_segments_on_max_overlap(
        true_segs,
        method_segs,
        segs_a_id_column="truth_segment_id",
        segs_b_id_column="method_segment_id",
        geometry_a_column="geometry_truth",
        geometry_b_column="geometry_method",
    )

    truth_to_nucleus = join_segments_on_max_overlap(
        true_segs,
        nuclear_segs,
        segs_a_id_column="truth_segment_id",
        segs_b_id_column="nuclear_segment_id",
        geometry_a_column="geometry_truth",
        geometry_b_column="geometry_nuclear",
    )

    results = (
        true_segs[["truth_segment_id"]]
        .merge(
            truth_to_method[
                ["truth_segment_id", "method_segment_id", "geometry_method"]
            ],
            left_on="truth_segment_id",
            right_on="truth_segment_id",
            how="left",
        )
        .merge(
            truth_to_nucleus[
                [
                    "truth_segment_id",
                    "nuclear_segment_id",
                    "geometry_truth",
                    "geometry_nuclear",
                ]
            ],
            left_on="truth_segment_id",
            right_on="truth_segment_id",
            how="left",
        )
        .replace({np.nan: None})
    )

    def get_union(row):
        if row["geometry_method"] is None:
            if (
                row["geometry_nuclear"] is not None
                and row["geometry_truth"] is not None
            ):
                return row["geometry_truth"].union(row["geometry_nuclear"]).area
            else:
                return np.nan
        else:
            if row["geometry_nuclear"] is not None:
                return (
                    row["geometry_truth"]
                    .union(row["geometry_method"].union(row["geometry_nuclear"]))
                    .area
                )
            elif row["geometry_truth"] is not None:
                return row["geometry_truth"].union(row["geometry_method"]).area
            else:
                return np.nan

    results["union"] = results.apply(lambda row: get_union(row), axis=1)

    def get_intersection(row):
        if row["geometry_method"] is None:
            if (
                row["geometry_nuclear"] is not None
                and row["geometry_truth"] is not None
            ):
                return row["geometry_truth"].intersection(row["geometry_nuclear"]).area
            else:
                return np.nan
        else:
            if row["geometry_nuclear"] is not None:
                return (
                    row["geometry_truth"]
                    .intersection(row["geometry_method"].union(row["geometry_nuclear"]))
                    .area
                )
            elif row["geometry_truth"] is not None:
                return row["geometry_truth"].intersection(row["geometry_method"]).area
            else:
                return np.nan

    results["intersection"] = results.apply(lambda row: get_intersection(row), axis=1)

    results["iou"] = results["intersection"] / results["union"]

    def get_jaccard_method_segment(row):
        if row["geometry_method"] is None:
            if row["geometry_nuclear"] is not None:
                return row["geometry_nuclear"]
            else:
                return box(0, 0, 0.001, 0.001)
        else:
            if row["geometry_nuclear"] is not None:
                return row["geometry_method"].union(row["geometry_nuclear"])
            else:
                return row["geometry_method"]

    def get_jaccard_truth_segment(row):
        if row["geometry_truth"] is None:
            if row["geometry_nuclear"] is not None:
                return row["geometry_nuclear"]
            else:
                return box(0, 0, 0.001, 0.001)
        else:
            return row["geometry_truth"]

    results["jaccard_method_segment"] = results.apply(
        lambda row: get_jaccard_method_segment(row), axis=1
    )

    results["jaccard_truth_segment"] = results.apply(
        lambda row: get_jaccard_truth_segment(row), axis=1
    )

    method_transcripts = spatial_join_polygons_and_transcripts(
        boundaries=results[["truth_segment_id", "jaccard_method_segment"]].set_geometry(
            "jaccard_method_segment"
        ),
        transcripts=transcripts_gdf,
    )

    truth_transcripts = spatial_join_polygons_and_transcripts(
        boundaries=results[["truth_segment_id", "jaccard_truth_segment"]].set_geometry(
            "jaccard_truth_segment"
        ),
        transcripts=transcripts_gdf,
    )

    segment_id_to_method_transcripts = (
        method_transcripts[["truth_segment_id", "index_right"]]
        .groupby("truth_segment_id")
        .agg({"index_right": set})["index_right"]
        .reset_index()
        .rename(columns={"index_right": "method_transcripts"})
    )

    segment_id_to_truth_transcripts = (
        truth_transcripts[["truth_segment_id", "index_right"]]
        .groupby("truth_segment_id")
        .agg({"index_right": set})["index_right"]
        .reset_index()
        .rename(columns={"index_right": "truth_transcripts"})
    )

    segment_id_to_transcripts = segment_id_to_truth_transcripts.merge(
        segment_id_to_method_transcripts,
        left_on="truth_segment_id",
        right_on="truth_segment_id",
    )

    segment_id_to_transcripts["jaccard_intersection"] = segment_id_to_transcripts.apply(
        lambda row: len(
            row["method_transcripts"].intersection(row["truth_transcripts"])
        ),
        axis=1,
    )

    segment_id_to_transcripts["jaccard_union"] = segment_id_to_transcripts.apply(
        lambda row: len(row["method_transcripts"].union(row["truth_transcripts"])),
        axis=1,
    )

    segment_id_to_transcripts["jaccard_index"] = (
        segment_id_to_transcripts["jaccard_intersection"]
        / segment_id_to_transcripts["jaccard_union"]
    )

    results = results.merge(
        segment_id_to_transcripts[
            [
                "truth_segment_id",
                "jaccard_intersection",
                "jaccard_union",
                "jaccard_index",
            ]
        ],
        left_on="truth_segment_id",
        right_on="truth_segment_id",
        how="left",
    )

    return results


def convert_transcripts_to_anndata(
    transcript_gdf,
    segmentation_gdf,
    gene_name_column="feature_name",
    min_molecules_per_cell=None,
):
    segmentation_gdf["area"] = segmentation_gdf.geometry.area
    segmentation_gdf["centroid_x"] = segmentation_gdf.geometry.centroid.x
    segmentation_gdf["centroid_y"] = segmentation_gdf.geometry.centroid.y
    if "index" in transcript_gdf.columns:
        del transcript_gdf["index"]
    if "index" in segmentation_gdf.columns:
        del segmentation_gdf["index"]

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_gdf, transcripts=transcript_gdf
    )
    sjoined_gdf.reset_index(inplace=True, drop=False, names="index")

    before_dedupe = len(sjoined_gdf)

    # if more than one row has the same index_right value, drop until index_right is unique
    sjoined_gdf = sjoined_gdf.drop_duplicates(subset="index_right")

    after_dedupe = len(sjoined_gdf)

    logger.info(
        f"Dropped {before_dedupe - after_dedupe} transcripts assigned to multiple segments"
    )

    # filter transcripts mapped to cell where the total number of transcripts for that cell is less than
    # min_molecules_per_cell
    if min_molecules_per_cell is not None:
        before_min_molecules = len(sjoined_gdf)

        sjoined_gdf = sjoined_gdf.groupby("index").filter(
            lambda x: len(x) >= min_molecules_per_cell
        )

        after_min_molecules = len(sjoined_gdf)
        logger.info(
            f"Dropped {before_min_molecules - after_min_molecules} cells with fewer than {min_molecules_per_cell} transcripts"
        )

    summed_counts_per_cell = (
        sjoined_gdf.groupby(["index", gene_name_column])
        .size()
        .reset_index(name="count")
    ).rename(columns={"index": "cell_id"})

    cell_u = list(sorted(summed_counts_per_cell["cell_id"].unique()))
    gene_u = list(sorted(transcript_gdf[gene_name_column].unique()))

    summed_counts_per_cell["cell_id_idx"] = pd.Categorical(
        summed_counts_per_cell["cell_id"], categories=cell_u, ordered=True
    )

    summed_counts_per_cell[gene_name_column] = pd.Categorical(
        summed_counts_per_cell[gene_name_column], categories=gene_u, ordered=True
    )

    data = summed_counts_per_cell["count"].tolist()
    row = summed_counts_per_cell["cell_id_idx"].cat.codes
    col = summed_counts_per_cell[gene_name_column].cat.codes

    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(cell_u), len(gene_u)))

    shapefile_index = summed_counts_per_cell["cell_id"].unique()
    shapefile_index.sort()

    additional_columns = [x for x in segmentation_gdf.columns if x != "geometry"]

    adata = anndata.AnnData(
        X=sparse_matrix,
        obsm={
            "spatial": segmentation_gdf.loc[shapefile_index][
                ["centroid_x", "centroid_y"]
            ].values
        },
        obs=segmentation_gdf.loc[shapefile_index][additional_columns],
        var=pd.DataFrame(index=gene_u),
    )

    adata.obs_names.name = "cell_id"

    return adata


def calculate_proportion_cyto_transcripts(
    transcript_gdf,
    segmentation_gdf,
    nuclei_gdf,
    segment_id_column=None,
):
    if segment_id_column is None:
        segment_id_column = "segment_id"
        segmentation_gdf = segmentation_gdf.reset_index(
            inplace=True, drop=False, names=segment_id_column
        )

    cytoplasm_shapes = segmentation_gdf.overlay(
        nuclei_gdf, how="difference", keep_geom_type=True
    )

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=cytoplasm_shapes, transcripts=transcript_gdf
    )

    cytoplasm_counts = (
        sjoined_gdf.groupby([segment_id_column])
        .size()
        .reset_index(name="cytoplasm_count")
    )

    sjoined_gdf = spatial_join_polygons_and_transcripts(
        boundaries=segmentation_gdf, transcripts=transcript_gdf
    )

    total_counts = (
        sjoined_gdf.groupby([segment_id_column])
        .size()
        .reset_index(name="segmentation_count")
    )

    return cytoplasm_counts.merge(
        total_counts, left_on=segment_id_column, right_on=segment_id_column
    )
