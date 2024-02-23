Inputs and Outputs
==================

Inputs
------

The expected input to nuc2seg is the Xenium analysis directory.

We expect a single directory with all files having the default names with no prefix or suffix
as described in the Xenium documentation here: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-at-a-glance

Specifically our algorithm uses the ``nucleus_boundaries.parquet`` and ``transcripts.parquet`` files, and we expect
them to be named as such.

Outputs
-------

nuc2seg produces several output data files:

- ``preprocessed.h5``: Rasterized transcript and nucleus data used for training the segmentation model.
- ``epoch=X-step=Y.ckpt``: Saved model weights (X will be the epoch number and Y will the step number where training ended)
- ``segmentation.h5``: A h5 file containing the rasterized segmentation of the cells.
- ``shapes.parquet``: A parquet file containing the non-rasterized shapes of the cell segmentation.
- ``anndata.h5ad``: Anndata file based on the cell segmentation (will only contain transcripts that fall within cell segments)
- ``spatialdata.zarr``: SpatialData Zarr directory with the cell segmentation added as a new layer, can be used for visualization with ``napari-spatialdata``

And several plots:


- ``prediction_plots/``: A directory containing plots of the model's angle predictions on the input data.
                         We create a hundred plots each covering a single tile just to allow for a quick visual inspection of the model's predictions.
- ``cell_typing_plots/``: A directory containing plots of the model's cell typing predictions on the input data.
                          Includes AIC, BIC for all k-values, as well as cell type probabilities and relative gene expression
                          for each k-value.
- ``segmentation.png``: A plot of the final segmentation. Red is the original nucleus, and blue is the predicted cell boundary.
- ``class_assignment.png``: A plot of the final cell typing. Each cell segment is colored according to its predicted cell type.
