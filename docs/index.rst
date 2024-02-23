nuc2seg
=======

Welcome to the documentation for the Python implementation of ``nuc2seg``

``nuc2seg`` is method for cell body segmentation of 10X Xenium data.

The default Xenium analysis output includes very accurate nucleus segmentation, but the cell body segmentation
uses a simple watershed algorithm that is not always accurate. ``nuc2seg`` solves the problem of cell body segmentation
by using the nucleus segmentation, along with the transcripts and rudimentary cell typing, to assign better cell body
segments.

``nuc2seg`` is provided as an nf-core nextflow pipeline. All standard nf-core pipeline features are available.
Read more about nf-core here: https://nf-co.re/docs/usage/introduction


Quickstart
----------

.. code::

        nextflow run tansey-lab/nuc2seg \
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
