process CREATE_SPATIALDATA {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/spatialdata:latest' :
        'docker.io/jeffquinnmsk/spatialdata:latest' }"

    input:
    tuple val(meta), path(segmentation), path(anndata), path(xenium_dir)

    output:
    tuple val(meta), path("${prefix}/spatialdata.zarr"), emit: zarr
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    create_sd \
        --segmentation ${segmentation} \
        --xenium-dir ${xenium_dir} \
        --output ${prefix}/tmpdir \
        ${args}

    zip -r -0 spatialdata.zarr ${prefix}/tmpdir
    rm -rf ${prefix}/tmpdir

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"

    echo create_sd.py \
        --segmentation ${segmentation} \
        --xenium-dir ${xenium_dir} \
        --anndata ${anndata} \
        --output-dir ${prefix}/tmpdir \
        ${args}

    touch ${prefix}/spatialdata.zarr
    touch versions.yml
    """
}
