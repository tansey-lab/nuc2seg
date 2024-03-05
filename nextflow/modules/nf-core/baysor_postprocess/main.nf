process BAYSOR_POSTPROCESS {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), path(shapefiles)

    output:
    tuple val(meta), path("${prefix}/baysor/segmentation.parquet"), emit: segmentation


    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/baysor"
    baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --output ${prefix}/baysor/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        ${args}
    """
}
