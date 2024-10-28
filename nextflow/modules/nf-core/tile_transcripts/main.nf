process TILE_TRANSCRIPTS {
    tag "$meta.id"
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), val(output_format)

    output:
    tuple val(meta), path("${prefix}/tiled_transcripts/*.${output_format}"), emit: transcripts, optional: true
    tuple val(meta), path("${prefix}/tiled_nuclei/*.${output_format}"), emit: transcripts, optional: true


    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    """
    mkdir -p "${prefix}/tiled_transcripts"
    tile_transcripts \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --output-dir ${prefix}/tiled_transcripts \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        --output-format ${output_format} \
        ${sample_area_arg} \
        ${args}
    """
}
