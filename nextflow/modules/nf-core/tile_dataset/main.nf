process TILE_DATASET {
    tag "$meta.id"
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(dataset), val(output_format)

    output:
    tuple val(meta), path("${prefix}/tiled_transcripts/*.${output_format}"), emit: transcripts, optional: true
    tuple val(meta), path("${prefix}/tiled_nuclei/*.${output_format}"), emit: nuclei, optional: true
    tuple val(meta), path("${prefix}/tiled_dataset/*.${output_format}"), emit: dataset, optional: true

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    """
    mkdir -p "${prefix}/tiled_transcripts"
    mkdir -p "${prefix}/tiled_nuclei"
    tile_dataset \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --transcript-output-dir ${prefix}/tiled_transcripts \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --nuclei-output-dir ${prefix}/tiled_nuclei \
        --dataset ${dataset} \
        --dataset-output-dir ${prefix}/tiled_dataset \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        --output-format ${output_format} \
        ${sample_area_arg} \
        ${args}
    """
}
