process SOPA_SEGMENT_BAYSOR {
    tag "$meta.id"
    label 'process_medium'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/sopa:' + params.sopa_version) :
        ('docker.io/jeffquinnmsk/sopa:' + params.sopa_version) }"

    input:
    tuple val(meta), path(sopa_zarr), path(patch_index_fn)

    output:
    tuple val(meta), env(PATCH_INDEX), emit: segments

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}"

    read -r -d '' PATCH_INDEX < ${patch_index_fn}

    cat << EOF > config.toml
    [data]
    force_2d = true
    min_molecules_per_cell = ${params.baysor_min_molecules_per_cell}
    x = "x"
    y = "y"
    z = "z"
    gene = "${params.gene_column_name}"
    min_molecules_per_gene = ${params.min_molecules_per_gene}
    min_molecules_per_segment = ${params.min_molecules_per_segment}
    confidence_nn_id = 6

    [segmentation]
    prior_segmentation_confidence = 1.0
    estimate_scale_from_centers = true
    n_clusters = ${params.baysor_n_clusters}
    iters = ${params.baysor_iters}
    n_cells_init = 0
    nuclei_genes = ""
    cyto_genes = ""
    EOF

    if [ ! -f "${sopa_zarr}/.sopa_cache/baysor_boundaries/\${PATCH_INDEX}.parquet" ]; then
        sopa segmentation baysor \
            --patch-index \$PATCH_INDEX \
            --config config.toml \
            ${sopa_zarr} \
            ${args}
    fi
    """
}
