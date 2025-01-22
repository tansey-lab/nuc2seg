process SOPA_PATCHIFY {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr)

    output:
    tuple val(meta), env(n_patches), emit: n_patches


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"
    sopa patchify image \
        --patch-width-pixel ${params.sopa_patch_pixel_size} \
        --patch-overlap-pixel 200 \
        ${sopa_zarr} \
        ${args}

    sopa patchify transcripts \
        --patch-width-microns ${params.sopa_patch_pixel_size} \
        --patch-overlap-microns 50 \
        --unassigned_value UNASSIGNED \
        --prior-shapes-key baysor_nuclear_prior \
        ${sopa_zarr} \
        ${args}

    n_patches=\$(cat "${sopa_zarr}/.sopa_cache/patches_file_image")
    """
}
