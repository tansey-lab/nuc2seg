process TRAIN {
    tag "$meta.id"
    label 'process_high'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/nuc2seg:latest' :
        'docker.io/jeffquinnmsk/nuc2seg:latest' }"

    input:
    tuple val(meta), path(dataset)

    output:
    tuple val(meta), path("${prefix}/*/checkpoints/*.ckpt"), emit: weights
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def wandb_key = params.wandb_api_key == null ? "" : "${params.wandb_api_key}"
    def wandb_mode = params.wandb_api_key == null ? "disabled" : "online"
    def runid = workflow.runName
    """
    export WANDB_DOCKER='jeffquinnmsk/nuc2seg:latest'
    mkdir -p "${prefix}"
    mkdir -p "${prefix}/wandb"
    export WANDB_DIR="\$(realpath ${prefix}/wandb)"
    export WANDB_DISABLE_GIT=1
    export WANDB_DISABLE_CODE=1
    export WANDB_API_KEY="${wandb_key}"
    export WANDB_MODE="${wandb_mode}"
    export WANDB_PROJECT="nuc2seg"
    export WANDB_RUN_ID="${runid}"
    train \
        --dataset ${dataset} \
        --output-dir ${prefix} \
        --tile-height ${params.tile_height} \
        --tile-width ${params.tile_width} \
        --overlap-percentage ${params.overlap_percentage} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        nuc2seg: \$( python -c 'from importlib.metadata import version;print(version("nuc2seg"))' )
    END_VERSIONS
    """
}
