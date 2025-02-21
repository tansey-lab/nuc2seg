from shapely import box
from nuc2seg.utils import (
    filter_anndata_to_sample_area,
    filter_anndata_to_min_transcripts,
    subset_anndata,
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
