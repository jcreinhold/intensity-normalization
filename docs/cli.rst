Executables
===========

Normalization scripts
---------------------

Z-Score Normalization
^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: zs_parser
   :prog: zscore-normalize

FCM Normalization
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: fcm_parser
   :prog: fcm-normalize

KDE Normalization
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: kde_parser
   :prog: kde-normalize

Piecewise Linear Histogram Matching (Nyul & Udupa)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: nyul_parser
   :prog: nyul-normalize

WhiteStripe Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: ws_parser
   :prog: ws-normalize

RAVEL Normalization
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: ravel_parser
   :prog: ravel-normalize

Least Squares Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: lsq_parser
   :prog: lsq-normalize

Other scripts
-------------

Plot Histograms
^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: histogram_parser
   :prog: plot-histograms

Preprocessing
^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: preprocess_parser
   :prog: preprocess

Tissue Mask
^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.exec.tissue_mask
   :func: arg_parser
   :prog: tissue-mask

Co-register
^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.cli
   :func: register_parser
   :prog: coregister
