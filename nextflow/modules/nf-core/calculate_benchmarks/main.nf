process CALCULATE_BENCHMARKS {
    tag "$meta.id"
    label 'process_high'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(segmentation_shapes), val(method_name)

    output:
    tuple val(meta), path("${prefix}/benchmark_results"), emit: results
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    calculate_benchmarks \
        --output-file ${prefix}/${method_name}_benchmarks.parquet \
        --segmentation-method-names ${method_name} \
        --segmentation-files ${segmentation_shapes} \
        --true-boundaries ${xenium_dir}/cell_boundaries.parquet \
        --nuclei-boundaries ${xenium_dir}/nucleus_boundaries.parquet \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --xenium-cell-metadata ${xenium_dir}/cells.parquet \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
