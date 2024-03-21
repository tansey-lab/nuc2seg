process BAYSOR {
    tag "$meta.id"
    label 'process_long'
    label 'process_high'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/baysor:latest' :
        'docker.io/jeffquinnmsk/baysor:latest' }"

    input:
    tuple val(meta), path(transcripts_file)

    output:
    tuple val(meta), path("${output_dir_name}/*.csv"), emit: segmentation
    tuple val(meta), path("${output_dir_name}/*_borders.html"), emit: plots
    tuple val(meta), path("${output_dir_name}/*_diagnostics.html"), emit: diagnostics
    tuple val(meta), path("${output_dir_name}/*_polygons.json"), emit: shapes

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def scale_arg = meta.baysor_scale == null ? "" : "--scale ${meta.baysor_scale}"
    def scale_std_arg = meta.baysor_scale_std == null ? "" : "--scale-std ${meta.baysor_scale_std}"
    def baysor_min_molecules_per_cell_arg = meta.baysor_min_molecules_per_cell == null ? "" : "--min-molecules-per-cell ${meta.baysor_min_molecules_per_cell}"
    def prior_segmentation_confidence_arg = meta.prior_segmentation_confidence == null ? "" : "--prior-segmentation-confidence ${meta.prior_segmentation_confidence}"
    def baysor_n_clusters_arg = meta.baysor_n_clusters == null ? "" : "--n-clusters ${meta.baysor_n_clusters}"
    output_dir_name = "${prefix}/baysor/min_molecules_per_cell=${meta.baysor_min_molecules_per_cell}/prior_segmentation_confidence=${meta.prior_segmentation_confidence}/baysor_scale=${meta.baysor_scale}/baysor_scale_std=${meta.baysor_scale_std}/baysor_n_clusters=${meta.baysor_n_clusters}"
    """
    mkdir -p "${output_dir_name}"
    output_fn=\$(basename -- "$transcripts_file")
    output_fn="\${output_fn%.*}"
    JULIA_NUM_THREADS=${task.cpus} baysor run \
        --x-column "x_location" \
        --y-column "y_location" \
        --z-column "z_location" \
        --gene-column "feature_name" \
        ${scale_arg} \
        ${scale_std_arg} \
        ${baysor_min_molecules_per_cell_arg} \
        ${prior_segmentation_confidence_arg} \
        ${baysor_n_clusters_arg} \
        --output ${output_dir_name}/\${output_fn}_result.csv \
        --plot \
        --save-polygons=GeoJSON \
        ${transcripts_file} :nucleus_id
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def scale_arg = meta.baysor_scale == null ? "" : "--scale ${meta.baysor_scale}"
    def scale_std_arg = meta.baysor_scale_std == null ? "" : "--scale-std ${meta.baysor_scale_std}"
    def baysor_min_molecules_per_cell_arg = meta.baysor_min_molecules_per_cell == null ? "" : "--min-molecules-per-cell ${meta.baysor_min_molecules_per_cell}"
    def prior_segmentation_confidence_arg = meta.prior_segmentation_confidence == null ? "" : "--prior-segmentation-confidence ${meta.prior_segmentation_confidence}"
    def baysor_n_clusters_arg = meta.baysor_n_clusters == null ? "" : "--n-clusters ${meta.baysor_n_clusters}"
    output_dir_name = "${prefix}/baysor/min_molecules_per_cell=${meta.baysor_min_molecules_per_cell}/prior_segmentation_confidence=${meta.prior_segmentation_confidence}/baysor_scale=${meta.baysor_scale}/baysor_scale_std=${meta.baysor_scale_std}/baysor_n_clusters=${meta.baysor_n_clusters}"
    """
    mkdir -p "${output_dir_name}"
    output_fn=\$(basename -- "$transcripts_file")
    output_fn="\${output_fn%.*}"
    echo JULIA_NUM_THREADS=${task.cpus} baysor run \
        --x-column "x_location" \
        --y-column "y_location" \
        --z-column "z_location" \
        --gene-column "feature_name" \
        ${scale_arg} \
        ${scale_std_arg} \
        ${baysor_min_molecules_per_cell_arg} \
        ${prior_segmentation_confidence_arg} \
        ${baysor_n_clusters_arg} \
        --output ${output_dir_name}/\${output_fn}_result.csv \
        --plot \
        --save-polygons=GeoJSON \
        ${transcripts_file} :nucleus_id

    touch ${output_dir_name}/\${output_fn}.csv
    touch ${output_dir_name}/\${output_fn}_borders.html
    touch ${output_dir_name}/\${output_fn}_diagnostics.html
    touch ${output_dir_name}/\${output_fn}_polygons.json
    """
}
