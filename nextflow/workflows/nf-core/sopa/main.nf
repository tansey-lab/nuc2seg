include { SOPA_EXTRACT_RESULT } from '../../../modules/nf-core/sopa_extract_result/main'
include { SOPA_EXTRACT_RESULT_BAYSOR } from '../../../modules/nf-core/sopa_extract_result_baysor/main'
include { SOPA_EXTRACT_RESULT_STARDIST } from '../../../modules/nf-core/sopa_extract_result_stardist/main'
include { SOPA_PATCHIFY_IMAGE } from '../../../modules/nf-core/sopa_patchify_image/main'
include { SOPA_PATCHIFY_TRANSCRIPTS } from '../../../modules/nf-core/sopa_patchify_transcripts/main'
include { SOPA_READ } from '../../../modules/nf-core/sopa_read/main'
include { SOPA_RESOLVE } from '../../../modules/nf-core/sopa_resolve/main'
include { SOPA_RESOLVE_BAYSOR } from '../../../modules/nf-core/sopa_resolve_baysor/main'
include { SOPA_RESOLVE_STARDIST } from '../../../modules/nf-core/sopa_resolve_stardist/main'
include { SOPA_SEGMENT } from '../../../modules/nf-core/sopa_segment/main'
include { SOPA_SEGMENT_BAYSOR } from '../../../modules/nf-core/sopa_segment_baysor/main'
include { SOPA_SEGMENT_STARDIST } from '../../../modules/nf-core/sopa_segment_stardist/main'
include { CALCULATE_BENCHMARKS } from '../../../modules/nf-core/calculate_benchmarks/main'

def create_parallel_sequence(meta, n_par) {
    def output = []
    n_par = n_par.toInteger()

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, x))
    }
    return output
}

workflow SOPA {
    def name = params.name == null ? "nuc2seg" : params.name

    ch_input = Channel.fromList([
        tuple( [ id: name, single_end:false ],
        file(params.xenium_dir, checkIfExists: true))
    ])

    if (params.zarr_file == null) {
        SOPA_READ( ch_input )
        SOPA_READ.out.zarr.tap { sopa_read_output }
    } else {
        sopa_read_output = Channel.fromList([
            tuple( [ id: name, single_end:false ], file(params.zarr_file, checkIfExists: true))
        ])
    }

    // Cellpose

    SOPA_PATCHIFY_IMAGE( sopa_read_output )

    SOPA_PATCHIFY_IMAGE.out.n_patches.flatMap { create_parallel_sequence(it[0], it[1]) }.tap { sopa_image_patches }

    sopa_segment_input = sopa_read_output.combine( sopa_image_patches, by: 0 )

    SOPA_SEGMENT( sopa_segment_input )

    sopa_read_output.join( SOPA_SEGMENT.out.segments.groupTuple() ).tap { sopa_resolve_input }

    SOPA_RESOLVE( sopa_resolve_input )

    ch_input.join( SOPA_RESOLVE.out.zarr ).tap { sopa_extract_result_input }

    SOPA_EXTRACT_RESULT( sopa_extract_result_input )

    SOPA_EXTRACT_RESULT.out.shapes.map { tuple(it[0], it[1], "cellpose") }.tap { cellpose_results }

    // Stardist

    SOPA_SEGMENT_STARDIST( sopa_segment_input )

    sopa_read_output.join( SOPA_SEGMENT_STARDIST.out.segments.groupTuple() ).tap { sopa_resolve_stardist_input }

    SOPA_RESOLVE_STARDIST( sopa_resolve_stardist_input )

    ch_input.join( SOPA_RESOLVE_STARDIST.out.zarr ).tap { sopa_extract_stardist_result_input }

    SOPA_EXTRACT_RESULT_STARDIST( sopa_extract_stardist_result_input )

    SOPA_EXTRACT_RESULT_STARDIST.out.shapes.map { tuple(it[0], it[1], "stardist") }.tap { stardist_results }

    // Baysor

    SOPA_PATCHIFY_TRANSCRIPTS( sopa_read_output )

    SOPA_PATCHIFY_TRANSCRIPTS.out.tx_patches.transpose().tap { sopa_tx_patches }

    sopa_segment_baysor_input = sopa_read_output.combine( sopa_tx_patches, by: 0 )

    SOPA_SEGMENT_BAYSOR( sopa_segment_baysor_input )

    sopa_read_output.join( SOPA_SEGMENT_BAYSOR.out.segments.groupTuple() ).tap { sopa_resolve_baysor_input }

    SOPA_RESOLVE_BAYSOR( sopa_resolve_baysor_input )

    ch_input.join( SOPA_RESOLVE_BAYSOR.out.zarr ).tap { sopa_extract_baysor_result_input }

    SOPA_EXTRACT_RESULT_BAYSOR( sopa_extract_baysor_result_input )

    SOPA_EXTRACT_RESULT_BAYSOR.out.shapes.map { tuple(it[0], it[1], "baysor") }.tap { baysor_results }

    // Calculate benchmarks
    concat(cellpose_results, stardist_results baysor_results).view()
    ch_input.join( concat(cellpose_results, stardist_results, baysor_results) ).tap { calculate_benchmarks_input }

    CALCULATE_BENCHMARKS( calculate_benchmarks_input )
}
