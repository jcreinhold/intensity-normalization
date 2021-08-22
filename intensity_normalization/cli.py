# -*- coding: utf-8 -*-
"""
intensity_normalization.cli

main functions and parsers for
all CLIs in intensity-normalization

** see parse.py for the CLI mix-in class! **
the mix-in class is placed in parse.py to
avoid circular imports.

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 07, 2021
"""

__all__ = [
    "fcm_main",
    "fcm_parser",
    "histogram_main",
    "histogram_parser",
    "kde_main",
    "kde_parser",
    "lsq_main",
    "lsq_parser",
    "nyul_main",
    "nyul_parser",
    "tissue_main",
    "tissue_parser",
    "ws_main",
    "ws_parser",
    "zs_main",
    "zs_parser",
]

from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.kde import KDENormalize
from intensity_normalization.normalize.lsq import LeastSquaresNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.zscore import ZScoreNormalize
from intensity_normalization.plot.histogram import HistogramPlotter
from intensity_normalization.util.tissue_membership import TissueMembershipFinder

fcm_parser = FCMNormalize.parser()
fcm_main = FCMNormalize.main(fcm_parser)

kde_parser = KDENormalize.parser()
kde_main = KDENormalize.main(kde_parser)

ws_parser = WhiteStripeNormalize.parser()
ws_main = WhiteStripeNormalize.main(ws_parser)

zs_parser = ZScoreNormalize.parser()
zs_main = ZScoreNormalize.main(zs_parser)

lsq_parser = LeastSquaresNormalize.parser()
lsq_main = LeastSquaresNormalize.main(lsq_parser)

nyul_parser = NyulNormalize.parser()
nyul_main = NyulNormalize.main(nyul_parser)

histogram_parser = HistogramPlotter.parser()
histogram_main = HistogramPlotter.main(histogram_parser)

tissue_parser = TissueMembershipFinder.parser()
tissue_main = TissueMembershipFinder.main(tissue_parser)


try:
    import ants
except (ModuleNotFoundError, ImportError):
    # ANTsPy not installed. Not loading preprocessor, co-registration, or RAVEL.
    pass
else:
    __all__ += [
        "preprocessor_main",
        "preprocessor_parser",
        "ravel_main",
        "ravel_parser",
        "register_main",
        "register_parser",
    ]

    from intensity_normalization.normalize.ravel import RavelNormalize
    from intensity_normalization.util.coregister import Registrator
    from intensity_normalization.util.preprocess import Preprocessor

    ravel_parser = RavelNormalize.parser()
    ravel_main = RavelNormalize.main(ravel_parser)

    preprocessor_parser = Preprocessor.parser()
    preprocessor_main = Preprocessor.main(preprocessor_parser)

    register_parser = Registrator.parser()
    register_main = Registrator.main(register_parser)
