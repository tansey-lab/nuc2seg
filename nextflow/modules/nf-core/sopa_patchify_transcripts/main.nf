process SOPA_PATCHIFY_TRANSCRIPTS {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr)

    output:
    tuple val(meta), path("${prefix}/transcript_tile_id_*"), emit: tx_patches


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    // Parse integer and divide by 0.21
    def micron_size = params.sopa_patch_pixel_size.toFloat() * 0.2125

    """
    mkdir -p "${prefix}"

    sopa patchify transcripts \
        --patch-width-microns ${micron_size} \
        --patch-overlap-microns 50 \
        --unassigned-value UNASSIGNED \
        --prior-shapes-key baysor_nuclear_prior \
        ${sopa_zarr} \
        ${args}

    awk '{print \$0 > ("transcript_tile_id_" (NR-1))}' "${sopa_zarr}/.sopa_cache/patches_file_transcripts"
    """
}
