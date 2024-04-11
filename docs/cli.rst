.. _cli:

Command Line Interface
======================

``nuc2seg`` is built as a series of command line tools that are invoked in sequence by nextflow.

You can use this reference to see which options are available for each command, and you can provide
these options with a custom nextflow config file. For example if you wanted to pass a custom number
of dataloaders to the predict step, you could add the following to your nextflow config file:

.. code::

    process {
        withName: PREDICT {
            ext.args = [
                '--num-dataloader-workers', '4'
            ].join(' ')
        }
    }


.. _cli_celltyping:

``celltyping``
--------------

.. argparse::
   :ref: nuc2seg.cli.celltyping.get_parser
   :prog: celltyping

.. _cli_preprocess:

``preprocess``
--------------

.. argparse::
   :ref: nuc2seg.cli.preprocess.get_parser
   :prog: preprocess


.. _cli_segment:

``segment``
-----------

.. argparse::
   :ref: nuc2seg.cli.segment.get_parser
   :prog: segment


.. _cli_train:

``train``
---------

.. argparse::
   :ref: nuc2seg.cli.train.get_parser
   :prog: train

.. _cli_predict:

``predict``
-----------

.. argparse::
   :ref: nuc2seg.cli.predict.get_parser
   :prog: predict

.. _cli_plot_predictions:

``plot_predictions``
--------------------

.. argparse::
   :ref: nuc2seg.cli.plot_predictions.get_parser
   :prog: plot_predictions
