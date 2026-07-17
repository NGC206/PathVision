"""Kokoro speech synthesis wrapper with a priority-based, preemptive queue."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class KokoroSpeaker:
    """Load Kokoro once and provide a priority-based, non-blocking speak API.

    Critical navigation messages (e.g. STOP, DANGER) immediately preempt
    and interrupt any ongoing normal guidance speech.
    """

    def __init__(
        self,
        enabled: bool,
        voice: str,
        language_code: str,
        speed: float,
        sample_rate: int,
        max_queue_size: int = 32,
    ) -> None:
        self._enabled = enabled
        self._voice = voice
        self._language_code = language_code
        self._speed = speed
        self._sample_rate = sample_rate
        
        self._pipeline: Any | None = None
        self._sounddevice: Any | None = None
        
        # Priority queue: elements are tuples of (priority_level, timestamp, text_to_speak)
        # Priority level 0 for critical alerts, 1 for normal guidance.
        self._queue: queue.PriorityQueue[tuple[int, float, str | None]] = queue.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.is_speaking = False
        self._last_enqueued_text = ""
        self._last_enqueued_ts = 0.0
        self._stop_requested = False

    def warmup(self) -> None:
        """Initialize Kokoro pipeline and start the background playback thread once."""
        if not self._enabled:
            return
            
        with self._lock:
            if self._pipeline is None:
                try:
                    from kokoro import KPipeline
                    import sounddevice as sd

                    self._pipeline = KPipeline(lang_code=self._language_code)
                    self._sounddevice = sd
                    LOGGER.info("Kokoro pipeline and sounddevice successfully loaded.")
                    
                    # Start the background worker thread
                    self._running = True
                    self._worker_thread = threading.Thread(
                        target=self._worker_loop, 
                        name="PathVisionSpeechWorker", 
                        daemon=True
                    )
                    self._worker_thread.start()
                    LOGGER.info("Kokoro speech worker thread started.")
                except Exception as exc:
                    LOGGER.exception("Failed to initialize Kokoro pipeline: %s", exc)

    def speak(self, text: str, blocking: bool = False, priority: int = 2) -> None:
        """Enqueue guidance text to be spoken with a priority (0 to 4).

        
        Priority 0 and 1 events immediately preempt and purge lower priority events.
        """
        if not self._enabled or not text.strip():
            return
            
        self.warmup()
        if self._pipeline is None:
            return

        is_critical = (priority <= 1)
        LOGGER.info("Speech request enqueued (priority=%d): '%s'", priority, text)

        # Suppress repeated non-critical utterances within a short window.
        now = time.time()
        if not is_critical and text.strip().lower() == self._last_enqueued_text and (now - self._last_enqueued_ts) < 4.0:
            LOGGER.debug("Suppressed repeated non-critical speech: '%s'", text)
            return

        if is_critical:
            LOGGER.info("Critical alert (priority %d) preempting speech queue.", priority)
            # 1. Request preemption on the worker thread safely
            self._stop_requested = True

            # 2. Clear all lower priority messages (priority >= 2) from the queue
            temp_list: list[tuple[int, float, str | None]] = []
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    # Retain only other high-priority alerts
                    if item[0] <= 1:
                        temp_list.append(item)
                except queue.Empty:
                    break
            # Put retained critical alerts back
            for item in temp_list:
                self._queue.put(item)

        # Enqueue the new message, dropping stale non-critical queue items if full.
        if self._queue.full() and not is_critical:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put((priority, now, text))
        self._last_enqueued_text = text.strip().lower()
        self._last_enqueued_ts = now

        if blocking:
            # Wait until the queue is fully processed
            self._queue.join()

    def _worker_loop(self) -> None:
        """Background loop to process speech items from the priority queue."""
        while self._running:
            try:
                # Blocks until an item is available
                priority, _, text = self._queue.get()
                
                # Check for shutdown sentinel
                if text is None:
                    self._queue.task_done()
                    break

                self._play_text(text)
                self._queue.task_done()
            except Exception as exc:
                LOGGER.exception("Error in speech worker loop: %s", exc)
                time.sleep(0.1)

    def _play_text(self, text: str) -> None:
        """Synthesize text and play it through sounddevice."""
        if self._pipeline is None or self._sounddevice is None:
            return

        self.is_speaking = True
        LOGGER.info("Speech playback started: '%s'", text)
        try:
            self._stop_requested = False
            generator = self._pipeline(text, voice=self._voice, speed=self._speed)
            for _, _, audio in generator:
                if not self._running or self._stop_requested:
                    break
                
                # Play audio sample
                self._sounddevice.play(audio, self._sample_rate)
                
                # Polling sleep loop to check for preemption (prevents PortAudio MME driver crash on Windows)
                duration = len(audio) / self._sample_rate
                t_start = time.perf_counter()
                while time.perf_counter() - t_start < duration:
                    if not self._running or self._stop_requested:
                        break
                    time.sleep(0.05)
            LOGGER.info("[QWEN] Speech completed: '%s'", text)
        except Exception as exc:
            LOGGER.error("Kokoro speech synthesis/playback failed: %s", exc)
        finally:
            self.is_speaking = False

    def stop_speech(self) -> None:
        """Halt all active sound playback and clear the normal speech queue."""
        self._stop_requested = True
        LOGGER.info("Preemption requested for active speech.")
        # Clear normal queue items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        LOGGER.info("Speech queue cleared.")

    def close(self) -> None:
        """Gracefully stop and clean up the speech worker thread and sound device."""
        LOGGER.info("Shutting down Kokoro Speaker...")
        self._running = False
        
        # Stop any active playback immediately
        if self._sounddevice is not None:
            try:
                self._sounddevice.stop()
            except Exception as exc:
                LOGGER.warning("Error stopping sounddevice during shutdown: %s", exc)

        # Clear queue and inject shutdown sentinel
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        self._queue.put((0, time.time(), None))
        
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            LOGGER.info("Kokoro speaker shutdown complete.")
