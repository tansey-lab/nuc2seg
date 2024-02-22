process SEGMENT {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(dataset), path(predictions), path(xenium_dir)

    output:
    tuple val(meta), path("${prefix}/segmentation.h5"), emit: segmentation
    tuple val(meta), path("${prefix}/shapes.parquet") , emit: shapefile
    tuple val(meta), path("${prefix}/anndata.h5ad")   , emit: anndata
    tuple val(meta), path("${prefix}/*.png")          , emit: plot
    path  "versions.yml"                              , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    segment \
        --output ${prefix}/segmentation.h5 \
        --shapefile-output ${prefix}/shapes.parquet \
        --anndata-output ${prefix}/anndata.h5ad \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --dataset ${dataset} \
        --predictions ${predictions} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
