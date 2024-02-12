import geopandas
from shapely import box
from matplotlib import pyplot as plt


def plot_tiling(bboxes):
    polygons = [box(*x) for x in bboxes]
    gdf = geopandas.GeoDataFrame(geometry=polygons)

    fig, ax = plt.subplots()

    gdf.boundary.plot(ax=ax, color="red", alpha=0.5)

    fig.savefig("/tmp/tiling.pdf")
