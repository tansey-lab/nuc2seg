process SEGMENT {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(xenium_dir), path(dataset), path(predictions), path(cell_typing_results), val(tile_idx)

    output:
    tuple val(meta), path("${prefix}/segmentation_tiles/segmentation_tile_*.h5") , emit: segmentation, optional: true
    tuple val(meta), path("${prefix}/shape_tiles/shapes_tile_*.parquet")         , emit: shapefile, optional: true
    tuple val(meta), path("${prefix}/anndata_tiles/anndata_tile_*.h5ad")         , emit: anndata, optional: true
    path  "versions.yml"                                                         , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    def n_celltypes_arg = params.n_celltypes == null ? "" : "--n-celltypes ${params.n_celltypes}"
    """
    mkdir -p "${prefix}/segmentation_tiles"
    mkdir -p "${prefix}/shape_tiles"
    mkdir -p "${prefix}/anndata_tiles"

    segment \
        --output ${prefix}/segmentation_tiles/segmentation_tile_${tile_idx}.h5 \
        --shapefile-output ${prefix}/shape_tiles/shapes_tile_${tile_idx}.parquet \
        --anndata-output ${prefix}/anndata_tiles/anndata_tile_${tile_idx}.h5ad \
        --transcripts ${xenium_dir}/transcripts.parquet \
        --celltyping-results ${cell_typing_results} \
        --dataset ${dataset} \
        --predictions ${predictions} \
        --tile-index ${tile_idx} \
        --tile-height ${params.segmentation_tile_height} \
        --tile-width ${params.segmentation_tile_width} \
        --overlap-percentage ${params.overlap_percentage} \
        ${n_celltypes_arg} \
        ${sample_area_arg} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
