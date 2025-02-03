process SOPA_EXTRACT_RESULT {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(sopa_zarr)

    output:
    tuple val(meta), path("${prefix}/cellpose_shapes.parquet"), emit: shapes

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    extract_shapefile_from_sopa \
        --output "${prefix}/cellpose_shapes.parquet" \
        --xenium-experiment "${xenium_dir}/experiment.xenium" \
        --zarr ${sopa_zarr} \
        --shapes-key cellpose_boundaries \
        ${args}
    """
}
