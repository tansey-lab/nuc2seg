include { PREPROCESS } from '../../../modules/nf-core/preprocess/main'
include { TRAIN } from '../../../modules/nf-core/train/main'
include { PREDICT } from '../../../modules/nf-core/predict/main'
include { PLOT_PREDICTIONS } from '../../../modules/nf-core/plot_predictions/main'
include { SEGMENT } from '../../../modules/nf-core/segment/main'
include { CREATE_SPATIALDATA } from '../../../modules/nf-core/create_spatialdata/main'
include { CELLTYPING } from '../../../modules/nf-core/celltyping/main'


def create_parallel_sequence(meta, fn, n_par) {
    def output = []

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, fn, x, n_par))
    }
    return output
}


workflow NUC2SEG {
    def name = params.name == null ? "nuc2seg" : params.name

    ch_input = Channel.fromList([
        tuple( [ id: name, single_end:false ], file(params.xenium_dir, checkIfExists: true), params.celltyping_n_chains)
    ])

    if (params.weights == null && params.dataset == null) {
        ch_input.flatMap { create_parallel_sequence(it[0], it[1], it[2]) }.tap { cell_typing_input }
        CELLTYPING( cell_typing_input )

        ch_input.map { tuple(it[0], it[1]) }.join(CELLTYPING.out.cell_typing_results.groupTuple()).tap { preprocess_input }

        PREPROCESS ( preprocess_input )
        TRAIN( PREPROCESS.out.dataset )

        PREPROCESS.out.dataset
            .join(TRAIN.out.weights)
            .tap { predict_input }
    } else {
        if (params.dataset != null && params.weights == null) {
            train_input = Channel.fromList([tuple( [ id: name, single_end:false ],
                  file(params.dataset, checkIfExists: true)
            )])
            TRAIN( PREPROCESS.out.dataset )
            train_input
                .join(TRAIN.out.weights)
                .tap { predict_input }
        } else {
            predict_input = Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
              file(params.dataset, checkIfExists: true),
              file(params.weights, checkIfExists: true)
            )])
        }
    }

    PREDICT( predict_input )

    if (params.dataset == null) {
        PREPROCESS.out.dataset
            .join(PREDICT.out.predictions)
            .join(ch_input.map { tuple(it[0], it[1]) })
            .tap { segment_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true)
        )])
            .join(PREDICT.out.predictions)
            .join(ch_input.map { tuple(it[0], it[1]) })
            .tap { segment_input }
    }

    SEGMENT( segment_input )

    SEGMENT.out.shapefile
        .join(SEGMENT.out.anndata)
        .join(ch_input.map { tuple(it[0], it[1]) })
        .set{ create_spatialdata_input }

    CREATE_SPATIALDATA( create_spatialdata_input )

    if (params.dataset == null) {
        PREPROCESS.out.dataset
            .join(PREDICT.out.predictions)
            .join(SEGMENT.out.segmentation)
            .tap { plot_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ], // meta map
          file(params.dataset, checkIfExists: true)
        )])
            .join(PREDICT.out.predictions)
            .join(SEGMENT.out.segmentation)
            .tap { plot_input }
    }

    PLOT_PREDICTIONS( plot_input )
}
