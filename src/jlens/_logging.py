# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in stderr logging helper for the ``jlens`` logger."""

from __future__ import annotations

import logging
import time


def _human_duration(seconds: float) -> str:
    """Format a duration: ``4s``, ``12m34s``, ``2h15m``, ``1d04h``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


class _DeltaFormatter(logging.Formatter):
    """Adds ``%(elapsed)s`` (since construction) and ``%(delta)s`` (since the
    previous record) to log records."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._t0 = self._last = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        now = time.monotonic()
        record.elapsed = f"{_human_duration(now - self._t0):>7}"
        record.delta = f"+{now - self._last:6.2f}s"
        self._last = now
        return super().format(record)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a stderr handler with ``[elapsed +delta] message`` formatting to
    the ``jlens`` logger. Idempotent. For scripts and notebooks; library
    callers that already configure :mod:`logging` should not call this."""
    package_logger = logging.getLogger("jlens")
    package_logger.setLevel(level)
    if any(isinstance(h, logging.StreamHandler) for h in package_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_DeltaFormatter("[%(elapsed)s %(delta)s] %(message)s"))
    package_logger.addHandler(handler)
    package_logger.propagate = False
