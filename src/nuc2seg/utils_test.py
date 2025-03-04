import numpy as np
import torch
from shapely import box
from nuc2seg.utils import (
    filter_anndata_to_sample_area,
    filter_anndata_to_min_transcripts,
    subset_anndata,
    transform_shapefile_to_rasterized_space,
    reassign_angles_for_centroids,
)


def test_filter_anndata_to_sample_area(test_adata):
    filtered = filter_anndata_to_sample_area(test_adata, box(0, 0, 10, 10))
    assert len(filtered) == 1
    assert filtered.obs.nucleus_label[0] == 1


def test_filter_anndata_to_min_transcripts(test_adata):
    filtered = filter_anndata_to_min_transcripts(test_adata, 2)
    assert len(filtered) == 2


def test_subset_anndata(test_adata):
    filtered = subset_anndata(test_adata, 1)
    assert len(filtered) == 1

    filtered = subset_anndata(test_adata, 100)
    assert len(filtered) == 2


def test_transform_shapefile_to_rasterized_space(test_nuclei_df):
    result = transform_shapefile_to_rasterized_space(
        gdf=test_nuclei_df, sample_area=(5, 5, 20, 20), resolution=0.5
    )
    assert len(result) == 1
    assert result.geometry.tolist()[0].area == 4.0


def test_reassign_angles_for_centroids():
    labels = torch.tensor(
        [
            [2, 2, 0, 0, 0],
            [2, 2, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [-1, -1, 0, 0, 0],
        ]
    )

    reassign_angles_for_centroids(labels, np.array([[2, 2], [0, 0]]))
