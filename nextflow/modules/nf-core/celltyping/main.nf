process CELLTYPING {
    tag "$meta.id"
    label 'process_medium'
    cpus 1
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), val(chain_idx), val(n_chains)

    output:
    tuple val(meta), path("${prefix}/cell_typing_chain_${chain_idx}.h5"), emit: cell_typing_results
    path  "versions.yml"                                                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    celltyping \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --transcripts-file ${xenium_dir}/transcripts.parquet \
        --output ${prefix}/cell_typing_chain_${chain_idx}.h5 \
        --n-chains ${n_chains} \
        --index ${chain_idx} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
