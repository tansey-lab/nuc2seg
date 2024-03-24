process BAYSOR_POSTPROCESS {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), path(shapefiles), path(transcript_assignments)

    output:
    tuple val(meta), path("${prefix}/baysor/segmentation.parquet"), emit: segmentation
    tuple val(meta), path("${prefix}/baysor/anndata.h5ad"), emit: anndata
    tuple val(meta), path("${prefix}/baysor/*.png"), emit: plots

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/baysor"
    baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --output ${prefix}/baysor/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/baysor"
    echo baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --output ${prefix}/baysor/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}
    touch ${prefix}/baysor/segmentation.parquet
    touch ${prefix}/baysor/segmentation.png
    touch ${prefix}/baysor/anndata.h5ad
    """
}
