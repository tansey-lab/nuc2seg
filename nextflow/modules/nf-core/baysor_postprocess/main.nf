process BAYSOR_POSTPROCESS {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(xenium_dir), path(shapefiles)

    output:
    tuple val(meta), path("${output_dir_name}/segmentation.parquet"), emit: segmentation
    tuple val(meta), path("${output_dir_name}/anndata.h5ad"), emit: anndata
    tuple val(meta), path("${output_dir_name}/*.png"), emit: plots

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    output_dir_name = "${prefix}/baysor/min_molecules_per_cell=${meta.baysor_min_molecules_per_cell}/prior_segmentation_confidence=${meta.prior_segmentation_confidence}/baysor_scale=${meta.baysor_scale}/baysor_scale_std=${meta.baysor_scale_std}/baysor_n_clusters=${meta.baysor_n_clusters}"

    """
    mkdir -p "${output_dir_name}"
    baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --output ${output_dir_name}/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    output_dir_name = "${prefix}/baysor/min_molecules_per_cell=${meta.baysor_min_molecules_per_cell}/prior_segmentation_confidence=${meta.prior_segmentation_confidence}/baysor_scale=${meta.baysor_scale}/baysor_scale_std=${meta.baysor_scale_std}/baysor_n_clusters=${meta.baysor_n_clusters}"

    """
    mkdir -p "${output_dir_name}"
    echo baysor_postprocess \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --nuclei-file ${xenium_dir}/nucleus_boundaries.parquet \
        --output ${output_dir_name}/segmentation.parquet \
        --baysor-shapefiles ${shapefiles} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}
    touch ${output_dir_name}/segmentation.parquet
    touch ${output_dir_name}/segmentation.png
    touch ${output_dir_name}/anndata.h5ad
    """
}
