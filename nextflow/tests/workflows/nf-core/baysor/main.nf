#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYSOR_SEGMENTATION } from '../../../../workflows/nf-core/baysor/main'

workflow baysor_workflow {
    BAYSOR_SEGMENTATION (  )
}
