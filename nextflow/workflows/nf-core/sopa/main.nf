include { SOPA_READ } from '../../../modules/nf-core/sopa_read/main'
include { SOPA_PATCHIFY } from '../../../modules/nf-core/sopa_patchify/main'
include { SOPA_SEGMENT } from '../../../modules/nf-core/sopa_segment/main'
include { SOPA_RESOLVE } from '../../../modules/nf-core/sopa_resolve/main'

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

    SOPA_PATCHIFY( sopa_read_output )

    SOPA_PATCHIFY.out.n_patches.flatMap { create_parallel_sequence(it[0], it[1]) }.tap { sopa_patches }

    sopa_segment_input = SOPA_READ.out.zarr.combine( sopa_patches, by: 0 )

    SOPA_SEGMENT( sopa_segment_input )

    SOPA_READ.out.zarr.join( SOPA_SEGMENT.out.segments.groupTuple() ).tap { sopa_resolve_input }

    SOPA_RESOLVE( sopa_resolve_input )
}
