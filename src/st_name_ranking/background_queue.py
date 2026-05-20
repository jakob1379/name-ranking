"""Background pair queue for responsive tournament voting."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Final

import streamlit as st

from st_name_ranking.pair_selection import PairSelectionOptions, select_candidate_batch

logger = logging.getLogger(__name__)

QUEUE_MANAGER_KEY: Final[str] = "st_name_ranking_queue_manager"
MIN_TRAINING_SAMPLES: Final[int] = 10
MIN_NAMES_FOR_PAIR_SELECTION: Final[int] = 2


class QueueManager:
    """Thread-safe background queue of tournament pairs."""

    def __init__(
        self,
        names: list[str],
        target_size: int = 15,
        refill_threshold: int = 5,
        sample_size: int = 50,
    ) -> None:
        if not names:
            raise ValueError("names list cannot be empty")
        if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
            raise ValueError(f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names")

        self.names: list[str] = names
        self.names_key: tuple[str, ...] = tuple(names)
        self.queue: deque[tuple[str, str]] = deque()
        self.target_size: int = max(target_size, 1)
        self.refill_threshold: int = max(refill_threshold, 1)
        self.sample_size: int = sample_size
        self.refill_count: int = 0
        self.last_refill_ms: float = 0.0
        self.avg_refill_ms: float = 0.0
        self.last_refill_added: int = 0
        self.last_refill_timestamp: float = 0.0
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background filler thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.debug("Background thread already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._fill_queue_continuously,
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Started QueueManager background thread (target_size=%d)", self.target_size)

    def stop(self) -> None:
        """Stop the background thread and wait for it to finish."""
        self._stop_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            if self._worker_thread.is_alive():
                logger.warning("Background thread did not stop within timeout")
            else:
                logger.info("Stopped QueueManager background thread")
        else:
            logger.debug("No background thread to stop")

    def get_pair(self) -> tuple[str, str] | None:
        """Pop the next queued pair without blocking."""
        with self._lock:
            if self.queue:
                pair = self.queue.popleft()
                logger.debug("Popped pair from queue, %d remaining", len(self.queue))
                return pair
            return None

    def get_queue_size(self) -> int:
        """Return the current queue size."""
        with self._lock:
            return len(self.queue)

    def get_stats(self) -> dict[str, int | float]:
        """Get queue and refill stats (thread-safe)."""
        with self._lock:
            return {
                "queue_size": len(self.queue),
                "target_size": self.target_size,
                "refill_threshold": self.refill_threshold,
                "sample_size": self.sample_size,
                "refill_count": self.refill_count,
                "last_refill_ms": self.last_refill_ms,
                "avg_refill_ms": self.avg_refill_ms,
                "last_refill_added": self.last_refill_added,
                "last_refill_timestamp": self.last_refill_timestamp,
            }

    def _fill_queue_continuously(self) -> None:
        """Keep the queue above the refill threshold."""
        logger.debug("Background filler thread started")

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    current_size = len(self.queue)

                if current_size < self.refill_threshold:
                    self._refill_queue()

                time.sleep(0.1)

            except Exception:
                logger.exception("Error in background queue filler")
                time.sleep(0.5)  # Longer sleep on error

        logger.debug("Background filler thread stopped")

    def _refill_queue(self) -> None:
        """Refill the queue with model-selected or random pairs."""
        import time

        start_time = time.perf_counter()
        logger.info("🔄 Queue refill started (current_size=%d, target=%d)", self.get_queue_size(), self.target_size)

        with self._lock:
            current_size = len(self.queue)
        needed = self.target_size - current_size

        if needed <= 0:
            return

        pairs = select_candidate_batch(
            self.names,
            options=PairSelectionOptions(
                batch_size=needed,
                sample_size=self.sample_size,
                min_training_samples=MIN_TRAINING_SAMPLES,
                fallback="random",
            ),
        )
        if not pairs:
            logger.error("Pair selection returned no pairs")
            return

        with self._lock:
            existing_pairs = set(self.queue)

            added = 0
            for pair in pairs:
                normalized = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                if normalized not in existing_pairs:
                    self.queue.append(pair)
                    existing_pairs.add(normalized)
                    added += 1

            elapsed = time.perf_counter() - start_time
            self.refill_count += 1
            self.last_refill_ms = elapsed * 1000
            self.avg_refill_ms += (self.last_refill_ms - self.avg_refill_ms) / self.refill_count
            self.last_refill_added = added
            self.last_refill_timestamp = time.time()
            if added > 0:
                logger.info("✅ Added %d pairs to queue (total: %d) in %.2fs", added, len(self.queue), elapsed)
            else:
                logger.info("⚠️ No new pairs added (all duplicates) in %.2fs", elapsed)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def __repr__(self) -> str:
        with self._lock:
            queue_size = len(self.queue)
        thread_status = "running" if self._worker_thread and self._worker_thread.is_alive() else "stopped"
        return f"QueueManager(target_size={self.target_size}, current_size={queue_size}, thread={thread_status})"


def get_queue_manager(
    names: list[str],
    target_size: int = 15,
    refill_threshold: int = 5,
    sample_size: int = 50,
) -> QueueManager:
    """Get or create the Streamlit session queue manager.

    sample_size controls the model-ranking subset used when refilling pairs.
    """
    names_key = tuple(names)
    if QUEUE_MANAGER_KEY in st.session_state:
        existing_manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]

        names_match = existing_manager.names_key == names_key
        size_match = existing_manager.target_size == target_size
        sample_size_match = existing_manager.sample_size == sample_size

        if names_match and size_match and sample_size_match:
            logger.debug("Reusing existing QueueManager from session state")
            return existing_manager

        logger.info(
            "Settings changed (names=%s, size=%s, sample_size=%s), stopping old QueueManager",
            not names_match,
            not size_match,
            not sample_size_match,
        )
        existing_manager.stop()
        del st.session_state[QUEUE_MANAGER_KEY]

    manager = QueueManager(
        names,
        target_size=target_size,
        refill_threshold=refill_threshold,
        sample_size=sample_size,
    )
    manager.start()
    st.session_state[QUEUE_MANAGER_KEY] = manager
    logger.info("Created and started new QueueManager (target_size=%d)", target_size)

    return manager


def get_queue_manager_stats() -> dict[str, int | float | str] | None:
    """Return stats for the current queue manager."""
    if QUEUE_MANAGER_KEY not in st.session_state:
        return None

    manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]
    stats: dict[str, int | float | str] = manager.get_stats()
    stats["num_names"] = len(manager.names)
    stats["thread_alive"] = manager._worker_thread is not None and manager._worker_thread.is_alive()
    return stats
