include { PREPROCESS } from '../../../modules/nf-core/preprocess/main'
include { TRAIN } from '../../../modules/nf-core/train/main'
include { PREDICT } from '../../../modules/nf-core/predict/main'
include { PLOT_PREDICTIONS } from '../../../modules/nf-core/plot_predictions/main'
include { SEGMENT } from '../../../modules/nf-core/segment/main'

workflow NUC2SEG {
    def name = params.name == null ? "nuc2seg" : params.name

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

    PREDICT( predict_input )

    PREPROCESS.out.dataset
        .join(PREDICT.out.predictions)
        .tap { plot_input }

    PLOT_PREDICTIONS( plot_input )

    PREPROCESS.out.dataset
        .join(PREDICT.out.predictions)
        .tap { segment_input }

    SEGMENT( segment_input )
}
