process TILE_DATASET {
    tag "$meta.id"
    label 'process_high_memory'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(dataset)

    output:
    tuple val(meta), path("${prefix}/tiled_dataset/*.h5"), emit: dataset, optional: true

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/tiled_dataset"
    tile_dataset \
        --dataset ${dataset} \
        --output-dir ${prefix}/tiled_dataset \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}
    """
}
