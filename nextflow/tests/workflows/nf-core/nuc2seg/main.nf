#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { NUC2SEG } from '../../../../workflows/nf-core/nuc2seg/main'

workflow nuc2seg_workflow {
    NUC2SEG (  )
}
