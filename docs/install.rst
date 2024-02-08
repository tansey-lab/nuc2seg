Installation
============

This tutorial will walk you through the process of setting up an environment
to run nuc2seg.

.. _install-nextflow-docker:

Option 1 (Recommended): Using Nextflow + Docker
-----------------------------------------------

nuc2seg uses several Python packages with C extensions,
so the easiest way to get started is using the up to date
docker image we maintain on docker hub.

.. code::

    docker pull jeffquinnmsk/nuc2seg:latest

To install Nextflow see the instructions here: https://www.nextflow.io/docs/latest/getstarted.html

Option 2: Install Using pip
---------------------------

For advanced usage, nuc2seg can be installed directly as a python package using pip.

.. code::

    pip install git+https://github.com/tansey-lab/nuc2seg
