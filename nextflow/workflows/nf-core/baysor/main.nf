include { BAYSOR } from '../../../modules/nf-core/baysor/main'
include { BAYSOR_PREPROCESS_TRANSCRIPTS } from '../../../modules/nf-core/baysor_preprocess_transcripts/main'
include { BAYSOR_POSTPROCESS } from '../../../modules/nf-core/baysor_postprocess/main'


workflow BAYSOR_SEGMENTATION {
    def name = params.name == null ? "nuc2seg" : params.name

    ch_input = Channel.fromList([
        tuple( [ id: name, single_end:false ], file(params.xenium_dir, checkIfExists: true))
    ])

    BAYSOR_PREPROCESS_TRANSCRIPTS( ch_input )

    baysor_min_molecules_per_cell_values = Channel.fromList( params.baysor_min_molecules_per_cell_values )
    prior_segmentation_confidence_values = Channel.fromList( params.prior_segmentation_confidence_values )
    baysor_scale_values = Channel.fromList( params.baysor_scale_values )
    baysor_scale_std_values = Channel.fromList( params.baysor_scale_std_values )
    baysor_n_clusters_values = Channel.fromList( params.baysor_n_clusters_values )

    baysor_min_molecules_per_cell_values
        .combine(prior_segmentation_confidence_values)
        .combine(baysor_scale_values)
        .combine(baysor_scale_std_values)
        .combine(baysor_n_clusters_values)
        .tap { baysor_param_sweep }


    BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts
        .combine( baysor_param_sweep )
        .tap { baysor_input }

    BAYSOR( baysor_input )

    ch_input.join(BAYSOR.out.shapes.groupTuple()).tap { postprocess_input }

    BAYSOR_POSTPROCESS( postprocess_input )
}
