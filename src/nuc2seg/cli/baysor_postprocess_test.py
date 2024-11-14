import json
import math
import os
import shutil
import tempfile
from unittest import mock

import anndata
import geopandas as gpd
import numpy as np
from blended_tiling import TilingModule

from nuc2seg.cli import baysor_postprocess
from nuc2seg.data import CelltypingResults
from nuc2seg.utils import generate_tiles


def test_baysor_postprocess(
    test_baysor_shapefile, test_transcripts_df, test_nucleus_boundaries
):
    tmpdir = tempfile.mkdtemp()
    shapefiles_fn = os.path.join(tmpdir, "baysor_results.json")
    output_fn = os.path.join(tmpdir, "segmentation.parquet")
    transcripts_fn = os.path.join(tmpdir, "transcripts.parquet")
    celltyping_fn = os.path.join(tmpdir, "celltyping_results.h5")

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

    celltyping_results = CelltypingResults(
        aic_scores=np.array([1, 2, 3]),
        bic_scores=np.array([1, 2, 3]),
        expression_profiles=[
            np.random.random((2, 2)),
            np.random.random((3, 2)),
            np.random.random((4, 2)),
        ],
        prior_probs=[
            np.array([0.5, 0.5]),
            np.array([0.33, 0.33, 0.33]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ],
        relative_expression=[
            np.random.random((2, 2)),
            np.random.random((3, 2)),
            np.random.random((4, 2)),
        ],
        min_n_components=2,
        max_n_components=4,
        gene_names=np.array(["gene1", "gene2"]),
    )
    celltyping_results.save_h5(celltyping_fn)

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
            "--celltyping-results",
            celltyping_fn,
            "--nucleus-overlap-threshold",
            "0.1",
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
