import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
import json
import logging

from nuc2seg.data import generate_tiles
from blended_tiling import TilingModule
from shapely import box

logger = logging.getLogger(__name__)


def stitch_shapes(shapes: list[gpd.GeoDataFrame], tile_size, base_size, overlap):
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

    all_shapes = []
    for (mask, shapes), bbox in zip(zip(tile_masks, shapes), bboxes):
        mask = mask.detach().cpu().numpy()
        mask = ~(mask < 1).astype(bool)

        # get the index of the upper left most true value
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

        # select only rows where box contains the shapes
        selection_vector = shapes.geometry.within(selection_box)
        all_shapes.append(shapes[selection_vector])

    return gpd.GeoDataFrame(pd.concat(all_shapes, ignore_index=True))


def read_baysor_results(shapes_fn, transcripts_fn) -> gpd.GeoDataFrame:
    with open(shapes_fn) as f:
        geojson_data = json.load(f)

    records = []
    for geometry in tqdm.tqdm(geojson_data["geometries"]):
        if len(geometry["coordinates"][0]) <= 4:
            logger.debug(
                f"Skipping cell with {len(geometry['coordinates'][0])} vertices"
            )
        polygon = shapely.Polygon(geometry["coordinates"][0])
        records.append({"geometry": polygon, "cell": geometry["cell"]})

    gdf = gpd.GeoDataFrame(records)

    meta_df = pd.read_csv(transcripts_fn, usecols=["cell", "cluster"])
    meta_df["cell_id"] = meta_df["cell"].apply(lambda x: int(x.split("-")[-1]))
    cell_to_cluster = meta_df[["cell_id", "cluster"]].drop_duplicates()

    result = gdf.merge(cell_to_cluster, left_on="cell", right_on="cell_id")
    del result["cell_id"]
    return result
