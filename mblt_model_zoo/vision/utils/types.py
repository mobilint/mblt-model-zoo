"""Deprecated module alias for :mod:`mblt_vision.utils.types`."""

import sys

from mblt_vision.utils import types as _standalone_types

sys.modules[__name__] = _standalone_types
