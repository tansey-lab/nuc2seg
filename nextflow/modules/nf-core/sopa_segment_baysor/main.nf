process SOPA_SEGMENT_BAYSOR {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr), val(patch_index)

    output:
    tuple val(meta), path("${sopa_zarr}/.sopa_cache/baysor_boundaries/${patch_index}.parquet"), emit: segments

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"

    cat << EOF > config.toml
    [data]
    force_2d = true
    min_molecules_per_cell = 10
    x = "x"
    y = "y"
    z = "z"
    gene = "genes"
    min_molecules_per_gene = 0
    min_molecules_per_segment = 3
    confidence_nn_id = 6

    [segmentation]
    prior_segmentation_confidence = 1.0
    estimate_scale_from_centers = true
    n_clusters = 4
    iters = 500
    n_cells_init = 0
    nuclei_genes = ""
    cyto_genes = ""
    EOF

    sopa segmentation baysor \
        --patch-index ${patch_index} \
        --config '"config.toml"' \
        ${sopa_zarr} \
        ${args}
    """
}
