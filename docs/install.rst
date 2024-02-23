Installation
============

This tutorial will walk you through the process of setting up an environment
to run nuc2seg.

.. _install-nextflow-docker:

Option 1 (Recommended): Using Nextflow + Containers
---------------------------------------------------

To install Nextflow see the instructions here: https://www.nextflow.io/docs/latest/getstarted.html

Ensure there is a container engine installed on your system.

In the HPC context this will likely be Singularity. On your local machine this will likely be Docker.

To install Docker see the instructions here: https://docs.docker.com/get-docker/


Option 2: Install Using pip
---------------------------

You should only need to install nuc2seg as a Python package if you are planning to make
contributions to nuc2seg, otherwise do not do this.

nuc2seg can be installed directly as a Python package using pip:

.. code::

    pip install git+https://github.com/tansey-lab/nuc2seg
