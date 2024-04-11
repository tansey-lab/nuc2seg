process PREDICT {
    tag "$meta.id"
    label 'process_medium'
    label 'gpu'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(dataset), path(model_weights)

    output:
    tuple val(meta), path("${prefix}/predictions.h5"), emit: predictions
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    predict \
        --output ${prefix}/predictions.h5 \
        --dataset ${dataset} \
        --model-weights ${model_weights} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
