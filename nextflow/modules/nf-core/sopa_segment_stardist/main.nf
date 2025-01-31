process SOPA_SEGMENT_STARDIST {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr), val(patch_index)

    output:
    tuple val(meta), path("${sopa_zarr}/.sopa_cache/cellpose_boundaries/${patch_index}.parquet"), emit: segments

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"

    if [ ! -f "${sopa_zarr}/.sopa_cache/stardist_patch/${patch_index}.parquet" ]; then
        mkdir -p /tmp/.keras

        if [ -d \$HOME/.keras ]; then
            cp -r \$HOME/.keras/* /tmp/.keras
        fi

        KERAS_HOME=/tmp/.keras sopa segmentation generic-staining \
            --method-name stardist_patch \
            --method-kwargs '{"model_type":"2D_versatile_fluo","prob_thresh":${params.stardist_prob_thresh},"nms_thresh":${params.nms_thresh}}' \
            --channels DAPI \
            --patch-index ${patch_index} \
            ${sopa_zarr} \
            ${args}
    fi
    """
}
