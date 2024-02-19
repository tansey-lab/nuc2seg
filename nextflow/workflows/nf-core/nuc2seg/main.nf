include { PREPROCESS } from '../../../modules/nf-core/preprocess/main'
include { TRAIN } from '../../../modules/nf-core/train/main'
include { PREDICT } from '../../../modules/nf-core/predict/main'
include { PLOT_PREDICTIONS } from '../../../modules/nf-core/plot_predictions/main'
include { SEGMENT } from '../../../modules/nf-core/segment/main'

workflow NUC2SEG {
    def name = params.name == null ? "nuc2seg" : params.name

    if (params.weights == null || params.dataset == null) {
        input = tuple( [ id: name, single_end:false ], // meta map
                  file(params.transcripts, checkIfExists: true),
                  file(params.boundaries, checkIfExists: true)
                )
        ch_input = Channel.fromList([input])
        PREPROCESS ( ch_input )
        TRAIN( PREPROCESS.out.dataset )

        PREPROCESS.out.dataset
            .join(TRAIN.out.weights)
            .tap { predict_input }
    } else {
        predict_input = Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true),
          file(params.weights, checkIfExists: true)
        )])
    }

    PREDICT( predict_input )

    if (params.weights == null || params.dataset == null) {
        PREPROCESS.out.dataset
            .join(PREDICT.out.predictions)
            .tap { plot_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true)
        )])
            .join(PREDICT.out.predictions)
            .tap { plot_input }
    }

    PLOT_PREDICTIONS( plot_input )


    if (params.weights == null || params.dataset == null) {
        PREPROCESS.out.dataset
            .join(PREDICT.out.predictions)
            .tap { segment_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true)
        )])
            .join(PREDICT.out.predictions)
            .tap { segment_input }
    }

    SEGMENT( segment_input )
}
