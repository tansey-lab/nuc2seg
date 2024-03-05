import geopandas as gpd
import numpy as np
import pandas as pd

from nuc2seg.data import generate_tiles
from blended_tiling import TilingModule
from shapely import box


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
