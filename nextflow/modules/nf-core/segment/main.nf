process SEGMENT {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), val(tile_idx), path(dataset), path(transcripts), path(predictions), path(cell_typing_results)

    output:
    tuple val(meta), path("${prefix}/segmentation_tile_*.h5")                    , emit: segmentation, optional: true
    tuple val(meta), path("${prefix}/shapes_tile_*.parquet")                     , emit: shapefile, optional: true
    tuple val(meta), path("${prefix}/anndata_tile_*.h5ad")                       , emit: anndata, optional: true
    path  "versions.yml"                                                         , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    segment \
        --output ${prefix}/segmentation_tile_${tile_idx}.h5 \
        --shapefile-output ${prefix}/shapes_tile_${tile_idx}.parquet \
        --anndata-output ${prefix}/anndata_tile_${tile_idx}.h5ad \
        --transcripts ${transcripts} \
        --celltyping-results ${cell_typing_results} \
        --dataset ${dataset} \
        --predictions ${predictions} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
