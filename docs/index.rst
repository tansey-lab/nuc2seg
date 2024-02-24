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
            -profile <docker/singularity/mskcc_iris/...> \
            --xenium_dir <path to xenium output> \
            --outdir <path to output> \
            --wandb_api_key <optional weights and bias api key for tracking UNet training>



Contents
========

.. toctree::
    :maxdepth: 2

    install
    inputs_and_outputs
    algorithm
    cli
    api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
