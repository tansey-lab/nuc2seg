.. _mskcc-iris:

Running on MSKCC Iris Cluster
=============================

The MSKCC Iris cluster is a high performance computing cluster with access to
GPUs that speed up neural net training and prediction.

To run on Iris, first you will need to install nextflow if you have not already.

.. code-block:: bash

    module load java/20.0.1
    wget -qO- https://get.nextflow.io > "${HOME}/nextflow"
    chmod +x "${HOME}/nextflow"


Next make free accounts on https://hub.docker.com/ and https://wandb.ai/

You will need to be logged into dockerhub to avoid rate limiting, and wandb
tracking of training is very important for debugging.

Get the wandb api key and your dockerhub username and token from the respective settings pages on each site.

Add the following lines to your `~/.bashrc` file on iris:

.. code-block:: bash

    module load java/20.0.1
    module load singularity/3.7.1
    export PATH="$PATH:$HOME"
    export NXF_SINGULARITY_CACHEDIR="${HOME}/images"
    export WANDB_API_KEY="<your wandb api key>"
    export SINGULARITY_DOCKER_USERNAME="<your dockerhub username>"
    export SINGULARITY_DOCKER_PASSWORD="<your dockerhub token>"
    export NXF_SINGULARITY_HOME_MOUNT=true

Run ``source ~/.bashrc`` to apply the changes.


When you run the nextflow pipeline, provide the ``-profile iris`` flag to use the iris profile.
This will submit all jobs to slurm and request the correct resources automatically.
