process GET_N_TILES {
    tag "$meta.id"
    label 'process_low'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(dataset), val(tile_dim), val(overlap_percentage)

    output:
    tuple val(meta), env(n_patches),  emit: n_tiles
    path  "versions.yml", emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    get_n_tiles \
        --output-file "${prefix}/n_tiles_${tile_dim}_${overlap_percentage}.txt" \
        --dataset ${dataset} \
        --tile-width ${tile_dim} \
        --tile-height ${tile_dim} \
        --overlap-percentage ${overlap_percentage} \
        ${args}


    n_patches=\$(cat "${prefix}/n_tiles_${tile_dim}_${overlap_percentage}.txt")

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
