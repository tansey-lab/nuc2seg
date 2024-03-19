process BAYSOR {
    tag "$meta.id"
    label 'process_long'
    label 'process_high'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/baysor:latest' :
        'docker.io/jeffquinnmsk/baysor:latest' }"

    input:
    tuple val(meta), path(transcripts_file), val(baysor_min_molecules_per_cell), val(prior_segmentation_confidence), val(baysor_scale), val(baysor_scale_std), val(baysor_n_clusters)

    output:
    tuple val(meta), val(output_dir_name), path("${output_dir_name}/segmentation.csv"), emit: segmentation
    tuple val(meta), val(output_dir_name), path("${output_dir_name}/segmentation_borders.html"), emit: plots
    tuple val(meta), val(output_dir_name), path("${output_dir_name}/segmentation_diagnostics.html"), emit: diagnostics
    tuple val(meta), val(output_dir_name), path("${output_dir_name}/segmentation_polygons.json"), emit: shapes

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def scale_arg = baysor_scale == null ? "" : "--scale ${baysor_scale}"
    def scale_std_arg = baysor_scale_std == null ? "" : "--scale-std ${baysor_scale_std}"
    def baysor_min_molecules_per_cell_arg = baysor_min_molecules_per_cell == null ? "" : "--min-molecules-per-cell ${baysor_min_molecules_per_cell}"
    def prior_segmentation_confidence_arg = prior_segmentation_confidence == null ? "" : "--prior-segmentation-confidence ${prior_segmentation_confidence}"
    output_dir_name = "${prefix}/baysor/min_molecules_per_cell=${baysor_min_molecules_per_cell}/prior_segmentation_confidence=${prior_segmentation_confidence}/baysor_scale=${baysor_scale}/baysor_scale_std=${baysor_scale_std}/baysor_n_clusters=${baysor_n_clusters}"
    """
    mkdir -p "${output_dir_name}"
    JULIA_NUM_THREADS=${task.cpus} baysor run \
        --x-column "x_location" \
        --y-column "y_location" \
        --z-column "z_location" \
        --gene-column "feature_name" \
        ${scale_arg} \
        ${scale_std_arg} \
        ${baysor_min_molecules_per_cell_arg} \
        ${prior_segmentation_confidence_arg} \
        --n-clusters ${baysor_n_clusters} \
        --output ${output_dir_name}/segmentation.csv \
        --plot \
        --save-polygons=GeoJSON \
        ${transcripts_file} :nucleus_id
    """
}
