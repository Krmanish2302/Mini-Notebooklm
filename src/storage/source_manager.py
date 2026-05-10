"""
source_manager.py  —  DEPRECATED: use StorageManager instead.

This file is kept only for import compatibility.  It re-exports StorageManager
under the legacy SourceManager name so any code that still imports SourceManager
continues to work without crashing.

Fix (Bug 9): SourceManager was a dead duplicate of StorageManager with
divergent logic that was never imported by any pipeline.  Rather than delete
the file (which could break any future imports), we alias it to StorageManager
and emit a deprecation warning.
"""
from __future__ import annotations

import warnings

from .storage_manager import StorageManager


class SourceManager(StorageManager):
    """
    Deprecated alias for StorageManager.
    Use StorageManager directly for all new code.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SourceManager is deprecated and will be removed in a future release. "
            "Use StorageManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
