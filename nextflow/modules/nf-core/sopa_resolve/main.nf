process SOPA_RESOLVE {
    tag "$meta.id"
    label 'process_high'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr), path(segments)

    output:
    tuple val(meta), path("${sopa_zarr}/shapes/cellpose_boundaries/shapes.parquet"), emit: shapes
    tuple val(meta), path("${sopa_zarr}"), emit: zarr

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    sopa resolve cellpose \
        ${sopa_zarr} \
        ${args}
    """
}
