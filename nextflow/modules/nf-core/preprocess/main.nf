process PREPROCESS {
    tag "$meta.id"
    label 'process_high'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(cell_typing_results)

    output:
    tuple val(meta), path("${prefix}/preprocessed.h5")                  , emit: dataset
    tuple val(meta), path("${prefix}/cell_typing_plots/*.pdf")          , emit: plot
    path  "versions.yml"                                                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    """
    mkdir -p "${prefix}"
    preprocess \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --transcripts-file ${xenium_dir}/transcripts.parquet \
        --output ${prefix}/preprocessed.h5 \
        --celltyping-results ${cell_typing_results} \
        ${sample_area_arg} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
