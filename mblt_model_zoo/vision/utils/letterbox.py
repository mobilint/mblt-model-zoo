"""Deprecated module alias for :mod:`mblt_vision.utils.letterbox`."""

import sys

from mblt_vision.utils import letterbox as _standalone_letterbox

sys.modules[__name__] = _standalone_letterbox
