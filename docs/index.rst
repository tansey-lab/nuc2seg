nuc2seg
=======

Welcome to the documentation for the Python implementation of ``nuc2seg``

``nuc2seg`` is method for cell body segmentation of 10X Xenium data.

The default Xenium analysis includes very accurate nucleus segmentation via DAPI staining,
but there is no comparable cell body segmentation included. The cell body segmentation
that is provided uses a simple expansion of the nuclear segments that is not always accurate (
https://kb.10xgenomics.com/hc/en-us/articles/11301491138317-How-does-Xenium-perform-cell-segmentation ).
``nuc2seg`` solves the problem of cell body segmentation by using information about the distribution
of transcripts and the gene expression profiles of cell types in the slide to determine better cell body segments.

``nuc2seg`` is provided as an nf-core compatible nextflow pipeline. All standard nf-core pipeline features are available.
Read more about nf-core here: https://nf-co.re/docs/usage/introduction


Quickstart
----------

.. code::

        nextflow run tansey-lab/nuc2seg \
            -r main \
            -profile <docker/singularity/iris/...> \
            --xenium_dir <path to xenium output> \
            --wandb_api_key <optional weights and bias api key for tracking UNet training> \
            --outdir /your/outdir \
            -w /your/outdir/nf

If running on MSKCC iris cluster, see the :ref:`Running on MSKCC Iris Cluster <mskcc-iris>` section for instructions.

The nextflow pipeline also provides the following optional parameters:

- ``--dataset``: Existing ``preprocessed.h5`` file to use, will skip preprocessing.
- ``--celltyping_results``: Existing celltyping results to use (i.e. ``--celltyping_results "/your/outdir/cell_typing_chain_*.h5"``).
- ``--weights``: Existing ``.ckpt`` to use, will skip neural net training.
- ``--resume_weights``: Existing ``.ckpt`` to use, will initialize model weights to this checkpoint and continue training.
- ``--sample_area``: Area of the slide to clip in bounding box format ``x1,y1,x2,y2`` (e.g. ``--sample_area "0,0,1000,1000"``).


Contents
========

.. toctree::
    :maxdepth: 2

    install
    mskcc_iris
    inputs_and_outputs
    algorithm
    cli
    Github <https://github.com/tansey-lab/nuc2seg>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
