process PLOT_PREDICTIONS {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(dataset), path(predictions), path(segmentation)

    output:
    tuple val(meta), path("${prefix}/prediction_plots"), emit: results
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    plot_predictions \
        --dataset ${dataset} \
        --predictions ${predictions} \
        --tile-height ${params.tile_height} \
        --tile-width ${params.tile_width} \
        --overlap-percentage ${params.overlap_percentage} \
        --output-dir ${prefix}/prediction_plots \
        --segmentation ${segmentation} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
