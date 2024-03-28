import os
import shutil
import tempfile
import json
import geopandas as gpd
from unittest import mock
import anndata
import pandas

from nuc2seg.data import generate_tiles, CelltypingResults, Nuc2SegDataset
from blended_tiling import TilingModule
import numpy as np
import math
from nuc2seg.cli import preprocess


def test_baysor_postprocess(
    test_baysor_shapefile, test_transcripts_df, test_nucleus_boundaries
):
    tmpdir = tempfile.mkdtemp()
    nuclei_fn = os.path.join(tmpdir, "nucleus_boundaries.parquet")
    transcripts_fn = os.path.join(tmpdir, "transcripts.parquet")
    celltyping_results_fn = os.path.join(tmpdir, "celltyping_results.parquet")
    output_fn = os.path.join(tmpdir, "preprocessed.h5")
    test_nucleus_boundaries.to_parquet(nuclei_fn)
    test_transcripts_df.to_parquet(transcripts_fn)

    celltyping_results = CelltypingResults(
        aic_scores=np.array([1, 2, 3]),
        bic_scores=np.array([1, 2, 3]),
        expression_profiles=[
            np.array([[1, 2], [1, 2]]),
            np.array([[1, 2], [1, 2], [1, 2]]),
            np.array([[1, 2], [1, 2], [1, 2], [1, 2]]),
        ],
        prior_probs=[
            np.array([0.5, 0.5]),
            np.array([0.33, 0.33, 0.33]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ],
        relative_expression=[
            np.array([[1, 2], [1, 2]]),
            np.array([[1, 2], [1, 2], [1, 2]]),
            np.array([[1, 2], [1, 2], [1, 2], [1, 2]]),
        ],
        min_n_components=2,
        max_n_components=4,
        gene_names=np.array(["gene1", "gene2"]),
    )

    celltyping_results.save_h5(celltyping_results_fn)

    command_line_arguments = [
        "preprocess",
        "--nuclei-file",
        nuclei_fn,
        "--transcripts",
        transcripts_fn,
        "--output",
        output_fn,
        "--celltyping-results",
        celltyping_results_fn,
    ]

    with mock.patch("sys.argv", command_line_arguments):
        try:
            preprocess.main()
            result = Nuc2SegDataset.load_h5(output_fn)

            assert result.n_classes == 2
            assert result.n_genes == 2
        finally:
            shutil.rmtree(tmpdir)
