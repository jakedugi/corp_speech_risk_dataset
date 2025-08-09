"""Minimal progress and ETA printer for CLI workflows.

This is a lightweight alternative to external progress bars, designed to keep
dependencies minimal. It prints a single updating line with elapsed time, ETA,
and rate information.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


@dataclass
class Progress:
    """Simple progress printer with ETA.

    Call `update(current, total, extra=...)` to refresh the line. Use `finish()`
    at the end to write a newline.
    """

    label: str
    total: int
    start_time: float = time.time()
    last_refresh: float = 0.0
    min_interval: float = 0.1  # seconds

    def update(
        self, current: int, total: Optional[int] = None, extra: str = ""
    ) -> None:
        if total is None:
            total = self.total
        now = time.time()
        if (now - self.last_refresh) < self.min_interval and current < total:
            return
        self.last_refresh = now
        elapsed = now - self.start_time
        rate = (current / elapsed) if elapsed > 0 and current > 0 else 0.0
        remaining = max(0.0, (total - current) / rate) if rate > 0 else 0.0
        pct = (100.0 * current / total) if total else 0.0
        line = (
            f"{self.label}: {current}/{total} ({pct:5.1f}%) | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"eta {_format_seconds(remaining)} | "
            f"rate {rate:.2f}/s"
        )
        if extra:
            line += f" | {extra}"
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
