# -*- coding: utf-8 -*-
"""
intensity-normalization.cli

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 07, 2021
"""

__all__ = [
    "fcm_main",
    "fcm_parser",
    "kde_main",
    "kde_parser",
    "lsq_main",
    "lsq_parser",
    "nyul_main",
    "nyul_parser",
    "ravel_main",
    "ravel_parser",
    "ws_main",
    "ws_parser",
    "zs_main",
    "zs_parser",
]

from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.kde import KDENormalize
from intensity_normalization.normalize.lsq import LeastSquaresNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.ravel import RavelNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.zscore import ZScoreNormalize
from intensity_normalization.plot.histogram import HistogramPlotter
from intensity_normalization.util.coregister import Registrator


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

ravel_parser = RavelNormalize.parser()
ravel_main = RavelNormalize.main(ravel_parser)

histogram_parser = HistogramPlotter.parser()
histogram_main = HistogramPlotter.main(histogram_parser)

register_parser = Registrator.parser()
register_main = Registrator.main(register_parser)
