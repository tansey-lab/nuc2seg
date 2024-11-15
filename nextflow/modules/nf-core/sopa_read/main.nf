process SOPA_READ {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(xenium_dir)

    output:
    tuple val(meta), path("${prefix}/sdata.zarr"), emit: zarr

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    sopa read \
        ${xenium_dir} \
        --sdata-path "${prefix}/sdata.zarr" \
        --technology xenium \
        ${args}
    """
}