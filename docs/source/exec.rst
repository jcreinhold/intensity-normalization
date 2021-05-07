Executables
===========

Normalization scripts
---------------------

Z-Score Normalization
^^^^^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.zscore_normalize
   :func: arg_parser
   :prog: zscore-normalize

FCM Normalization
^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.fcm_normalize
   :func: arg_parser
   :prog: fcm-normalize

GMM Normalization
^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.gmm_normalize
   :func: arg_parser
   :prog: gmm-normalize

KDE Normalization
^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.kde_normalize
   :func: arg_parser
   :prog: kde-normalize

Piecewise Linear Histogram Matching (Nyul & Udupa)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.nyul_normalize
   :func: arg_parser
   :prog: nyul-normalize

WhiteStripe Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.ws_normalize
   :func: arg_parser
   :prog: ws-normalize

RAVEL Normalization
^^^^^^^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.ravel_normalize
   :func: arg_parser
   :prog: ravel-normalize

Least Squares Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.exec.lsq_normalize
   :func: arg_parser
   :prog: lsq-normalize

Other scripts
-------------

Plot Histograms
^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.exec.plot_hists
   :func: arg_parser
   :prog: plot-hists

Preprocessing
^^^^^^^^^^^^^

.. argparse:: 
   :module: intensity_normalization.exec.preprocess
   :func: arg_parser
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
   :module: intensity_normalization.exec.coregister
   :func: arg_parser
   :prog: coregister

Plot Quality Metric
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: intensity_normalization.exec.norm_quality
   :func: arg_parser
   :prog: norm-quality

