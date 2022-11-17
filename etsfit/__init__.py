# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

from .etsfit import etsMAIN
from . import utilities, snPlotting, MCMC
__all__ = ["etsMAIN"]
__version__ = "0.1.0"
__author__ = 'Lindsey Gordon'