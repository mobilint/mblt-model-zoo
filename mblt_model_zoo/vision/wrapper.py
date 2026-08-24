"""Deprecated module alias for mblt_vision.wrapper."""

import sys

from mblt_vision import wrapper as _standalone_wrapper

sys.modules[__name__] = _standalone_wrapper
