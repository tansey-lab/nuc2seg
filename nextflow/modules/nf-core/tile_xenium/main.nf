process TILE_XENIUM {
    tag "$meta.id"
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), val(output_format)

    output:
    tuple val(meta), path("${prefix}/tiled_transcripts/*.${output_format}"), emit: transcripts, optional: true
    tuple val(meta), path("${prefix}/tiled_nuclei/*.${output_format}"), emit: nuclei, optional: true

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    """
    mkdir -p "${prefix}/tiled_transcripts"
    mkdir -p "${prefix}/tiled_nuclei"
    tile_xenium \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --transcript-output-dir ${prefix}/tiled_transcripts \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --nuclei-output-dir ${prefix}/tiled_nuclei \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        --output-format ${output_format} \
        ${sample_area_arg} \
        ${args}
    """
}
