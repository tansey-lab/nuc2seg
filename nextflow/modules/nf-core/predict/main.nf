process PREDICT {
    tag "$meta.id"
    label 'process_medium'
    label 'gpu'
    container "${ workflow.containerEngine == 'apptainer' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) :
        ('docker.io/jeffquinnmsk/nuc2seg:' + params.nuc2seg_version) }"

    input:
    tuple val(meta), path(dataset), path(model_weights), val(job_index), val(n_jobs)

    output:
    tuple val(meta), path("${prefix}/predictions/thread_${job_index}.pt"), emit: predictions
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir -p "${prefix}/predictions"
    predict \
        --output-file "${prefix}/predictions/thread_${job_index}.pt" \
        --dataset ${dataset} \
        --model-weights ${model_weights} \
        --tile-width ${params.tile_width} \
        --tile-height ${params.tile_height} \
        --overlap-percentage ${params.overlap_percentage} \
        --n-jobs ${n_jobs} \
        --job-index ${job_index} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
