"""Stress-testing script to run the real PathVision pipeline, trigger scans, and monitor resources."""

import logging
import time
import sys
import threading
import traceback
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import load_config
from main import PathVisionApplication
from runtime.event_bus import Event, EventPriority, EventType
from reasoning.situation_manager import InteractionMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "stress_test.log", mode="w", encoding="utf-8")
    ]
)
LOGGER = logging.getLogger("stress_test")


def main():
    # Limit PyTorch CPU threads to prevent core saturation during Kokoro speech synthesis
    torch.set_num_threads(2)
    LOGGER.info("Starting PathVision Stress Test...")
    config = load_config()
    
    # Run headlessly to prevent GUI blocking/display server issues
    object.__setattr__(config.runtime, "show_preview", False)
    
    app = PathVisionApplication(config)
    
    # Thread to trigger periodic scans
    def scan_trigger():
        time.sleep(15.0)  # Wait for startup warmups to settle
        triggers = [
            (30.0, InteractionMode.ORIENTATION, "Environment Scan 1"),
            (45.0, InteractionMode.DESCRIPTION, "Deep Scan 1"),
            (60.0, InteractionMode.ORIENTATION, "Environment Scan 2"),
            (75.0, InteractionMode.DESCRIPTION, "Deep Scan 2"),
        ]
        
        start_time = time.perf_counter()
        for delay, mode, label in triggers:
            # Wait until the target elapsed time
            while time.perf_counter() - start_time < delay:
                if not app.manager._running:
                    return
                time.sleep(0.5)
            
            LOGGER.info("STRESS TEST: Triggering %s (mode=%s)...", label, mode.value)
            app.manager.event_bus.publish(Event(
                priority=EventPriority.COGNITIVE,
                event_type=EventType.SCAN_REQUEST,
                payload={"mode": mode}
            ))
            
        # Wait until the end of the test duration (15s startup + 75s scan trigger + 45s final navigation = 135s total)
        time.sleep(45.0)
        LOGGER.info("STRESS TEST: Reached end of 2-minute duration. Stopping manager...")
        app.manager.stop()

    # Thread to monitor resources
    def resource_monitor():
        start_time = time.perf_counter()
        stats_file = PROJECT_ROOT / "logs" / "stress_test_resources.csv"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("elapsed_sec,cpu_pct,ram_mb,vram_mb,speech_q\n")
            
        while app.manager._running:
            elapsed = time.perf_counter() - start_time
            
            if app.manager.health_monitor is not None:
                stats = app.manager.health_monitor.get_stats()
                
                cpu = stats.get("cpu_percent", 0.0)
                ram = stats.get("ram_used_mb", 0.0)
                vram = stats.get("gpu_mem_used_mb", 0.0)
                speech_q = stats.get("speech_queue_size", 0)
                
                LOGGER.info(
                    "STRESS TEST STATS: Elapsed: %.1fs | CPU: %.1f%% | RAM: %.1f MB | VRAM: %.1f MB | Speech Q: %d",
                    elapsed, cpu, ram, vram, speech_q
                )
                
                with open(stats_file, "a", encoding="utf-8") as f:
                    f.write(f"{elapsed:.1f},{cpu:.1f},{ram:.1f},{vram:.1f},{speech_q}\n")
            else:
                LOGGER.info("STRESS TEST: Waiting for health monitor initialization...")
                
            time.sleep(5.0)

    # Start triggering and monitoring threads
    t_trigger = threading.Thread(target=scan_trigger, name="TestTrigger", daemon=True)
    t_monitor = threading.Thread(target=resource_monitor, name="TestMonitor", daemon=True)
    
    t_trigger.start()
    t_monitor.start()

    # Run application
    try:
        app.start()
        # Wait for the duration (130 seconds)
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 130.0:
            if not app.manager._running:
                LOGGER.error("STRESS TEST: Runtime manager stopped early!")
                break
            time.sleep(1.0)
    except Exception as exc:
        LOGGER.error("STRESS TEST CRASHED: %s", exc)
        with open(PROJECT_ROOT / "logs" / "stress_test_crash.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
    finally:
        LOGGER.info("STRESS TEST: Stopping application...")
        app.manager.stop()
        LOGGER.info("STRESS TEST: Completed.")


if __name__ == "__main__":
    main()
