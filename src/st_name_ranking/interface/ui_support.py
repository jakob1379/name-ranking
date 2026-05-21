"""Shared support for Streamlit UI renderers."""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_NAMES_FOR_COMPARISON = 2
MIN_NAMES_FOR_LANDSCAPE = 25
MIN_NON_NOISE_CLUSTERS = 2
FAST_REFILL_THRESHOLD_MS = 120
MODERATE_REFILL_THRESHOLD_MS = 300
SLOW_RENDER_THRESHOLD_MS = 100
MS_PER_SECOND = 1000
FILTER_SAVE_INTERVAL = 50
MAX_EXCLUDED_NAMES_DISPLAY = 100


@dataclass(frozen=True)
class RenderTimer:
    """Small timing helper for Streamlit fragments."""

    label: str
    start_time: float

    @classmethod
    def start(cls, label: str) -> "RenderTimer":
        return cls(label=label, start_time=time.perf_counter())

    def log(self, step: str) -> None:
        logger.debug(
            "%s [%s]: %.2fms",
            self.label,
            step,
            (time.perf_counter() - self.start_time) * MS_PER_SECOND,
        )
