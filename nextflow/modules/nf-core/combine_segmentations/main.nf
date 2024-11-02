process COMBINE_SEGMENTATIONS {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(dataset), path(segmentations), path(shapefiles), path(adatas)

    output:
    tuple val(meta), path("${prefix}/shapes.parquet")                            , emit: shapefile
    tuple val(meta), path("${prefix}/anndata.h5ad")                              , emit: anndata
    tuple val(meta), path("${prefix}/*.png")                                     , emit: plots
    path  "versions.yml"                                                         , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def sample_area_arg = params.sample_area == null ? "" : "--sample-area ${params.sample_area}"
    """
    mkdir -p "${prefix}"
    combine_segmentations \
        --output-dir "${prefix}" \
        --dataset ${dataset} \
        --segmentation-outputs ${segmentations} \
        --adatas ${adatas} \
        --shapes ${shapefiles} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        ${sample_area_arg} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
