"""Deprecated module alias for :mod:`mblt_vision.utils.results`."""

import sys

from mblt_vision.utils import results as _standalone_results

sys.modules[__name__] = _standalone_results
