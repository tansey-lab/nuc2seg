process BAYSOR_POSTPROCESS {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), path(shapefiles), path(transcript_assignments)

    output:
    tuple val(meta), path("${prefix}/baysor/segmentation.parquet"), emit: segmentation
    tuple val(meta), path("${prefix}/baysor/*.png"), emit: plots

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/baysor"
    baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --baysor-transcript-assignments ${transcript_assignments} \
        --output ${prefix}/baysor/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        ${args}
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/baysor"
    echo baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --baysor-transcript-assignments ${transcript_assignments} \
        --output ${prefix}/baysor/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        ${args}
    touch ${prefix}/baysor/segmentation.parquet
    touch ${prefix}/baysor/segmentation.png
    """
}
