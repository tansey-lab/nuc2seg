include { CALCULATE_BENCHMARKS } from '../../../modules/nf-core/calculate_benchmarks/main'

workflow BENCHMARK {
    def name = params.name == null ? "nuc2seg" : params.name

    ch_input = Channel.fromList([
        tuple(
            [ id: name, single_end:false ],
            file(params.xenium_dir, checkIfExists: true),
            file(params.nuc2seg_shapes, checkIfExists: true),
            file(params.baysor_shapes, checkIfExists: true),
            file(params.cellpose_shapes, checkIfExists: true),
        )

    ])

    CALCULATE_BENCHMARKS(ch_input)
}
