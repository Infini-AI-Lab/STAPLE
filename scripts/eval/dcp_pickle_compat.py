# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read DCP checkpoints whose ``.metadata`` was pickled by a newer Python.

Python 3.13 split ``pathlib`` into a package and moved the concrete classes to
``pathlib._local``. A DCP ``.metadata`` written under 3.13 therefore pickles
paths as ``pathlib._local.PosixPath``, and unpickling it on 3.12 dies with

    ModuleNotFoundError: No module named 'pathlib._local'; 'pathlib' is not a package

The classes are the same objects under both layouts, so aliasing the module name
is enough. Importing this module installs the alias; it is a no-op on 3.13+,
where the real module already exists.

Import it before any ``torch.distributed.checkpoint`` read::

    import dcp_pickle_compat  # noqa: F401
"""

from __future__ import annotations

import pathlib
import sys
import types


_ALIAS = "pathlib._local"
_NAMES = (
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "Path",
    "PosixPath",
    "WindowsPath",
)


def install() -> bool:
    """Alias ``pathlib._local`` onto ``pathlib``. Returns True if it was added."""
    if _ALIAS in sys.modules:
        return False
    try:  # Python 3.13+ has the real module.
        __import__(_ALIAS)
    except ImportError:
        pass
    else:
        return False

    shim = types.ModuleType(_ALIAS)
    for name in _NAMES:
        attribute = getattr(pathlib, name, None)
        if attribute is not None:
            setattr(shim, name, attribute)
    sys.modules[_ALIAS] = shim
    return True


install()
