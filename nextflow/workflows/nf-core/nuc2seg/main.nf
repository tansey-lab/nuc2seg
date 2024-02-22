include { PREPROCESS } from '../../../modules/nf-core/preprocess/main'
include { TRAIN } from '../../../modules/nf-core/train/main'
include { PREDICT } from '../../../modules/nf-core/predict/main'
include { PLOT_PREDICTIONS } from '../../../modules/nf-core/plot_predictions/main'
include { SEGMENT } from '../../../modules/nf-core/segment/main'
include { CREATE_SPATIALDATA } from '../../../modules/nf-core/create_spatialdata/main'

workflow NUC2SEG {
    def name = params.name == null ? "nuc2seg" : params.name

    xenium_input = Channel.fromList([
        tuple( [ id: name, single_end:false ], file(params.xenium_dir, checkIfExists: true))
    ])

    if (params.weights == null || params.dataset == null) {
        PREPROCESS ( xenium_input )
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
            .join(xenium_input)
            .tap { segment_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true)
        )])
            .join(PREDICT.out.predictions)
            .join(xenium_input)
            .tap { segment_input }
    }

    SEGMENT( segment_input )

    SEGMENT.out.segmentation
        .join(SEGMENT.out.shapefile)
        .join(SEGMENT.out.anndata)
        .join(xenium_input)
        .set{ create_spatialdata_input }

    CREATE_SPATIALDATA( create_spatialdata_input )
}
