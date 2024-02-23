.. _cli:

Command Line Interface
======================

Nuc2Seg provides a suite of command line utilities that allow users to script running the pipeline end to end.


.. _cli_plot_predictions:

``plot_predictions``
------------------

.. argparse::
   :ref: nuc2seg.cli.plot_predictions.get_parser
   :prog: plot_predictions


.. _cli_predict:

``predict``
-----------------------------

.. argparse::
   :ref: nuc2seg.cli.predict.get_parser
   :prog: predict


.. _cli_calculate_scores:

``preprocess``
---------------------

.. argparse::
   :ref: nuc2seg.cli.preprocess.get_parser
   :prog: preprocess


.. _cli_segment:

``segment``
---------------------

.. argparse::
   :ref: nuc2seg.cli.segment.get_parser
   :prog: segment


.. _cli_train:

``train``
----------------

.. argparse::
   :ref: nuc2seg.cli.train.get_parser
   :prog: train
