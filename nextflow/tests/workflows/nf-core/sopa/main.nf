#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SOPA } from '../../../../workflows/nf-core/sopa/main'

workflow sopa_workflow {
    SOPA (  )
}
