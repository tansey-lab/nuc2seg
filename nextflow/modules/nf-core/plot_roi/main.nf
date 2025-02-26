process PLOT_ROI {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(dataset), path(predictions), path(shapes), path(prior_shapes)

    output:
    tuple val(meta), path("${prefix}/roi_plots"), emit: results
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/roi_plots"
    plot_roi \
        --dataset ${dataset} \
        --predictions ${predictions} \
        --prior-segments ${prior_shapes} \
        --segments ${shapes} \
        --output-dir ${prefix}/roi_plots \
        --transcripts ${xenium_dir}/transcripts.parquet \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
