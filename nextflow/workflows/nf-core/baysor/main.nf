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


    BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts.transpose()
        .combine( baysor_param_sweep )
        .map { tuple([id: it[0].id,
                      baysor_min_molecules_per_cell: it[2],
                      prior_segmentation_confidence: it[3],
                      baysor_scale: it[4],
                      baysor_scale_std: it[5],
                      baysor_n_clusters: it[6]], it[1]) }
        .tap { baysor_input }

    BAYSOR( baysor_input )

    BAYSOR.out.shapes.groupTuple()
        .map { tuple([id: it[0].id], file(params.xenium_dir, checkIfExists: true), it[1])}
        .tap { postprocess_input }

    BAYSOR_POSTPROCESS( postprocess_input )
}
