import geopandas as gpd
import numpy as np
from blended_tiling import TilingModule


def tile_transcripts_to_csv(
    shapes: list[gpd.GeoDataFrame], tile_size, base_size, overlap
):

    tiler = TilingModule(
        tile_size=tile_size,
        tile_overlap=(overlap, overlap),
        base_size=base_size,
    )

    tile_masks = tiler.get_tile_masks()[:, 0, :, :]

    for mask, shapes in zip(tile_masks, shapes):
        mask = ~(mask < 1).astype(bool)

        # get the index of the upper left most true value
        x, y = np.where(mask)
        x_min, x_max = x.min(), x.max()
