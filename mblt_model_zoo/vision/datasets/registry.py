"""Deprecated module alias for :mod:`mblt_vision.datasets.registry`."""

import sys

from mblt_vision.datasets import registry as _standalone_registry

sys.modules[__name__] = _standalone_registry
