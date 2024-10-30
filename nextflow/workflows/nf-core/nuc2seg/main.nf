include { PREPROCESS } from '../../../modules/nf-core/preprocess/main'
include { TRAIN } from '../../../modules/nf-core/train/main'
include { PREDICT } from '../../../modules/nf-core/predict/main'
include { PLOT_PREDICTIONS } from '../../../modules/nf-core/plot_predictions/main'
include { SEGMENT } from '../../../modules/nf-core/segment/main'
include { CREATE_SPATIALDATA } from '../../../modules/nf-core/create_spatialdata/main'
include { CELLTYPING } from '../../../modules/nf-core/celltyping/main'
include { TILE_DATASET } from '../../../modules/nf-core/tile_dataset/main'
include { TILE_XENIUM } from '../../../modules/nf-core/tile_xenium/main'

include { COMBINE_SEGMENTATIONS } from '../../../modules/nf-core/combine_segmentations/main'

def create_parallel_sequence(meta, fn, n_par) {
    def output = []

    for (x in (0..(n_par-1))) {
        output.add(tuple(meta, fn, x, n_par))
    }
    return output
}

def extractTileNumber(filepath) {
   def matcher = filepath.fileName.toString() =~ /tile_(\d+)\.[^.]+$/
   return matcher.find() ? matcher[0][1].toInteger() : null
}

workflow NUC2SEG {
    def name = params.name == null ? "nuc2seg" : params.name

    ch_input = Channel.fromList([
        tuple( [ id: name, single_end:false ], file(params.xenium_dir, checkIfExists: true), params.celltyping_n_chains)
    ])

    if (params.celltyping_results != null) {
        celltyping_results = Channel.fromList([
            tuple( [ id: name, single_end:false ], file(params.celltyping_results, checkIfExists: true))
        ])
    }

    preprocess_input = Channel.empty()

    if (params.weights == null && params.dataset == null && params.resume_weights == null) {

        if (params.celltyping_results == null) {
            ch_input.flatMap { create_parallel_sequence(it[0], it[1], it[2]) }.tap { cell_typing_input }
            CELLTYPING( cell_typing_input )
            CELLTYPING.out.cell_typing_results.groupTuple().tap { celltyping_results }
            ch_input.map { tuple(it[0], it[1]) }.join(celltyping_results).tap { preprocess_input }
        } else {
            preprocess_input = Channel.fromList(
                [
                    tuple(
                        [ id: name, single_end:false ],
                        file(params.xenium_dir, checkIfExists: true),
                        file(params.celltyping_results, checkIfExists: true)
                    )
                ]
            )
        }

        PREPROCESS ( preprocess_input )
        TRAIN( PREPROCESS.out.dataset.map {tuple(it[0], it[1], [])} )

        PREPROCESS.out.dataset
            .flatMap { create_parallel_sequence(it[0], it[1], params.n_predict_jobs) }
            .join(TRAIN.out.weights)
            .map { tuple(it[0], it[1], it[4], it[2], it[3]) }
            .tap { predict_input }
    }
    else if (params.resume_weights != null) {
        train_input = Channel.fromList([tuple( [ id: name, single_end:false ],
            file(params.dataset, checkIfExists: true),
            file(params.resume_weights, checkIfExists: true),
        )])
        TRAIN( train_input )
        train_input
          .flatMap { create_parallel_sequence(it[0], it[1], params.n_predict_jobs) }
          .join(TRAIN.out.weights)
          .map { tuple(it[0], it[1], it[4], it[2], it[3]) }
          .tap { predict_input }
    }
    else {
        if (params.dataset != null && params.weights == null) {
            train_input = Channel.fromList([tuple( [ id: name, single_end:false ],
                  file(params.dataset, checkIfExists: true),
                  []
            )])
            TRAIN( train_input )

            train_input
                .flatMap { create_parallel_sequence(it[0], it[1], params.n_predict_jobs) }
                .join(TRAIN.out.weights)
                .map { tuple(it[0], it[1], it[4], it[2], it[3]) }
                .tap { predict_input }
        } else {
            Channel.fromList([tuple( [ id: name, single_end:false ],
                file(params.dataset, checkIfExists: true))])
              .flatMap { create_parallel_sequence(it[0], it[1], params.n_predict_jobs) }
              .join(
                Channel.fromList([tuple( [ id: name, single_end:false ],
                  file(params.weights, checkIfExists: true))])
              )
              .map { tuple(it[0], it[1], it[4], it[2], it[3]) }
              .tap { predict_input }
        }
    }
    PREDICT( predict_input )

    TILE_XENIUM( ch_input.map { tuple(it[0], it[1], "parquet")} )

    if (params.dataset == null ) {
        PREPROCESS.out.dataset
            .tap { tile_dataset_input }
    } else {
        tile_dataset_input = Channel.fromList([tuple( [ id: name, single_end:false ],
                  file(params.dataset, checkIfExists: true))])
    }

    TILE_DATASET( tile_dataset_input )

    TILE_XENIUM.out.transcripts.transpose().map {
        tuple(it[0], extractTileNumber(it[1]), it[1])
    }.tap { tiled_transcripts }

    TILE_DATASET.out.dataset.transpose().map {
        tuple(it[0], extractTileNumber(it[1]), it[1])
    }.tap { tiled_dataset }

    PREDICT.out.predictions.transpose().map {
        tuple(it[0], extractTileNumber(it[1]), it[1])
    }.tap { tiled_predictions }

    tiled_dataset
        .join(tiled_transcripts, by: [0,1])
        .join(tiled_predictions, by: [0,1])
        .tap { tiled_data }

    tiled_data.view()

    tiled_data
        .join(celltyping_results)
        .tap { segment_input }

    SEGMENT( segment_input )

    if (params.dataset == null) {
        PREPROCESS.out.dataset.join(
            SEGMENT.out.segmentation.groupTuple()
        ).join(
            SEGMENT.out.shapefile.groupTuple()
        ).join(
            SEGMENT.out.anndata.groupTuple()
        ).tap { combine_segmentations_input }
    } else {
        Channel.fromList([tuple( [ id: name, single_end:false ],
          file(params.dataset, checkIfExists: true)
        )]).join(
            SEGMENT.out.segmentation.groupTuple()
        ).join(
            SEGMENT.out.shapefile.groupTuple()
        ).join(
            SEGMENT.out.anndata.groupTuple()
        ).tap { combine_segmentations_input }
    }

    COMBINE_SEGMENTATIONS( combine_segmentations_input )

    COMBINE_SEGMENTATIONS.out.shapefile
        .join(SEGMENT.out.anndata)
        .join(ch_input.map { tuple(it[0], it[1]) })
        .set{ create_spatialdata_input }

    CREATE_SPATIALDATA( create_spatialdata_input )
}
