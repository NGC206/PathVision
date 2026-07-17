"""Health Monitor and Thread Watchdog for PathVision Runtime v2.0."""

from __future__ import annotations

import logging
import os
import time
import threading
import queue
from typing import Any, Callable, Dict

import psutil
import torch

LOGGER = logging.getLogger(__name__)


class WorkerDescriptor:
    """Descriptor holding metadata and restart handlers for runtime workers."""

    def __init__(self, name: str, start_fn: Callable[[], None], stop_fn: Callable[[], None]) -> None:
        self.name = name
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.last_heartbeat = time.perf_counter()
        self.restart_count = 0
        self.status = "HEALTHY"


class HealthMonitor:
    """Watchdog thread tracking hardware load and auto-recovering crashed workers in isolation."""

    def __init__(self, event_bus: Any, resource_manager: Any, scheduler: Any) -> None:
        self.eb = event_bus
        self.rm = resource_manager
        self.scheduler = scheduler

        self._workers: Dict[str, WorkerDescriptor] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stats: dict[str, Any] = {}

    def register_worker(self, name: str, start_fn: Callable[[], None], stop_fn: Callable[[], None]) -> None:
        """Register a worker thread and its restart/stop callbacks with the watchdog."""
        with self._lock:
            self._workers[name] = WorkerDescriptor(name, start_fn, stop_fn)
            LOGGER.info("Registered worker with HealthMonitor: %s", name)

    def heartbeat(self, name: str) -> None:
        """Record a worker check-in to confirm thread liveness."""
        with self._lock:
            if name in self._workers:
                self._workers[name].last_heartbeat = time.perf_counter()
                self._workers[name].status = "HEALTHY"

    def get_stats(self) -> dict[str, Any]:
        """Fetch the latest captured resource usage metrics."""
        with self._lock:
            return dict(self._stats)

    def start(self) -> None:
        """Start the watchdog monitor thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, name="HealthMonitorThread", daemon=True)
        self._thread.start()
        LOGGER.info("HealthMonitor watchdog started.")

    def stop(self) -> None:
        """Shutdown the monitor loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        LOGGER.info("HealthMonitor stopped.")

    def _monitor_loop(self) -> None:
        """Continuous watchdog checking heartbeats and polling resource limits."""
        while self._running:
            time.sleep(1.0)
            now = time.perf_counter()

            # 1. Heartbeat check & Auto-Restart Recovery
            dead_workers = []
            with self._lock:
                for name, worker in self._workers.items():
                    # If no heartbeat for more than 15.0 seconds, consider thread dead
                    if now - worker.last_heartbeat > 15.0:
                        worker.status = "DEAD"
                        dead_workers.append(worker)

            for worker in dead_workers:
                LOGGER.error("WATCHDOG: Worker '%s' missed heartbeats! Attempting restart...", worker.name)
                try:
                    # Stop the old worker thread context
                    worker.stop_fn()
                    
                    # Wait up to 5.0 seconds for the old thread to stop
                    t_stop_start = time.perf_counter()
                    while self.scheduler and self.scheduler._thread and self.scheduler._thread.is_alive() and time.perf_counter() - t_stop_start < 5.0:
                        time.sleep(0.1)
                        
                    if self.scheduler and self.scheduler._thread and self.scheduler._thread.is_alive():
                        LOGGER.warning("WATCHDOG: Old scheduler thread is still alive. Skipping restart to prevent concurrent CUDA context collision.")
                        # Reset heartbeat to prevent loop hammering
                        with self._lock:
                            worker.last_heartbeat = time.perf_counter()
                        continue
                        
                    # Start a fresh instance of the thread
                    worker.start_fn()
                    
                    with self._lock:
                        worker.last_heartbeat = time.perf_counter()
                        worker.restart_count += 1
                        worker.status = "RECOVERED"
                    LOGGER.info("WATCHDOG: Worker '%s' successfully restarted. (Restart count: %d)", worker.name, worker.restart_count)
                except Exception as exc:
                    LOGGER.exception("WATCHDOG: Failed to auto-recover worker '%s': %s", worker.name, exc)

            # 2. Poll Hardware Resource Stats
            try:
                cpu_p = psutil.cpu_percent()
                cpu_per_core = psutil.cpu_percent(percpu=True)
                
                # Fetch cpu freq safely
                cpu_freq_obj = psutil.cpu_freq()
                cpu_f = cpu_freq_obj.current if cpu_freq_obj else 0.0
                
                ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                
                gpu_mem = 0.0
                if torch.cuda.is_available():
                    free_v, total_v = torch.cuda.mem_get_info()
                    gpu_mem = (total_v - free_v) / (1024 * 1024)

                # Queue sizes
                speech_q = self.rm.speaker._queue.qsize() if self.rm.speaker else 0
                telemetry_q = self.scheduler.config.learning.enabled # approximation or fetched from manager
                
                with self._lock:
                    self._stats = {
                        "cpu_percent": cpu_p,
                        "per_core_cpu": cpu_per_core,
                        "cpu_freq": cpu_f,
                        "ram_used_mb": ram,
                        "gpu_mem_used_mb": gpu_mem,
                        "speech_queue_size": speech_q,
                        "timestamp": now,
                    }
            except Exception as e:
                LOGGER.debug("Error collecting health metrics: %s", e)
