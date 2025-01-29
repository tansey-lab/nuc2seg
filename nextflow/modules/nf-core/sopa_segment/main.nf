process SOPA_SEGMENT {
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

    if [ ! -f "${sopa_zarr}/.sopa_cache/cellpose_boundaries/${patch_index}.parquet" ]; then
        sopa segmentation cellpose \
            --channels DAPI \
            --diameter ${params.sopa_cellpose_diameter} \
            --patch-index ${patch_index} \
            ${sopa_zarr} \
            ${args}
    fi
    """
}
