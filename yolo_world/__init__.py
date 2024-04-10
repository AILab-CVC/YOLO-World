# Copyright (c) Tencent Inc. All rights reserved.
import importlib.metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = '0.0.0'


from .models import *  # noqa
from .datasets import *  # noqa
from .engine import *  # noqa
