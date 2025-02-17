process SOPA_RESOLVE_STARDIST {
    tag "$meta.id"
    label 'process_high'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr), path(segments)

    output:
    tuple val(meta), path("${sopa_zarr}/shapes/stardist_patch/shapes.parquet"), emit: shapes
    tuple val(meta), path("${sopa_zarr}"), emit: zarr

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"

    if [ ! -f "${sopa_zarr}/shapes/stardist_patch/shapes.parquet" ]; then
        sopa resolve generic \
            --method-name stardist_patch \
            ${sopa_zarr} \
            ${args}
    fi
    """
}
