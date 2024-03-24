import os
import shutil
import tempfile
import json
import geopandas as gpd
from unittest import mock
import anndata
import pandas

from nuc2seg.data import generate_tiles
from blended_tiling import TilingModule
import numpy as np
import math
from nuc2seg.cli import baysor_postprocess


def test_baysor_postprocess(
    test_baysor_shapefile, test_transcripts_df, test_nucleus_boundaries
):
    tmpdir = tempfile.mkdtemp()
    shapefiles_fn = os.path.join(tmpdir, "baysor_results.json")
    output_fn = os.path.join(tmpdir, "segmentation.parquet")
    transcripts_fn = os.path.join(tmpdir, "transcripts.parquet")

    nuclei_fn = os.path.join(tmpdir, "nucleus_boundaries.parquet")
    test_nucleus_boundaries.to_parquet(nuclei_fn)

    overlap = 0.1
    tile_size = (5, 5)

    x_extent = math.ceil(test_transcripts_df["x_location"].astype(float).max())
    y_extent = math.ceil(test_transcripts_df["y_location"].astype(float).max())
    base_size = (x_extent, y_extent)

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
    baysor_shapefiles = []
    for idx, bbox in enumerate(bboxes):
        fn = os.path.join(tmpdir, f"tile_{idx}_segmentation.json")
        with open(fn, "w") as f:
            json.dump(test_baysor_shapefile, f)
        baysor_shapefiles.append(fn)

    test_transcripts_df.to_parquet(transcripts_fn)

    command_line_arguments = (
        ["baysor_postprocess", "--baysor-shapefiles"]
        + baysor_shapefiles
        + [
            "--nuclei-file",
            nuclei_fn,
            "--transcripts",
            transcripts_fn,
            "--output",
            output_fn,
            "--tile-width",
            f"{tile_size[0]}",
            "--tile-height",
            f"{tile_size[1]}",
            "--overlap-percentage",
            f"{overlap}",
        ]
    )

    with mock.patch("sys.argv", command_line_arguments):
        try:
            baysor_postprocess.main()
            anndata.read_h5ad(os.path.join(tmpdir, "anndata.h5ad"))
            assert os.path.exists(os.path.join(tmpdir, "segmentation.png"))
            gpd.read_parquet(output_fn)

        finally:
            shutil.rmtree(tmpdir)
