"""Latency, FPS, and VRAM benchmarking tool for PathVision v2.0."""

import os
import sys
import time
from pathlib import Path
import psutil
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from research.train_v2 import PathVisionSegModel

def run_benchmark():
    print("Initializing PathVision v2.0 Benchmarking Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = PathVisionSegModel().to(device)
    model.eval()
    
    # Warmup runs
    dummy_input = torch.randn(1, 3, 240, 320, device=device)
    print("Running warmup inference iterations...")
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    # Benchmarking runs
    print("Executing 200 benchmark iterations...")
    inf_times = []
    
    t_start_all = time.perf_counter()
    for _ in range(200):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_end = time.perf_counter()
        inf_times.append((t_end - t_start) * 1000.0)
        
    t_end_all = time.perf_counter()
    
    # Compute stats
    mean_latency = np.mean(inf_times)
    min_latency = np.min(inf_times)
    max_latency = np.max(inf_times)
    std_latency = np.std(inf_times)
    
    fps = 200.0 / (t_end_all - t_start_all)
    
    # RAM / CPU profiling
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024.0 * 1024.0)
    cpu_pct = process.cpu_percent(interval=0.1)
    
    # VRAM profiling
    if device.type == "cuda":
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        vram_max_allocated_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    else:
        vram_allocated_mb = 0.0
        vram_max_allocated_mb = 0.0
        
    print("\n==================================================")
    print("PATHVISION V2.0 BENCHMARK METRICS REPORT")
    print("==================================================")
    print(f"Mean Latency:          {mean_latency:.2f} ms")
    print(f"Min / Max Latency:     {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"Latency Std Dev:       {std_latency:.2f} ms")
    print(f"Pipeline Frame Rate:   {fps:.2f} FPS")
    print(f"Host CPU Usage:        {cpu_pct:.1f}%")
    print(f"Host RAM Footprint:    {ram_usage_mb:.2f} MB")
    if device.type == "cuda":
        print(f"GPU VRAM Allocated:    {vram_allocated_mb:.2f} MB")
        print(f"GPU VRAM Max Peek:     {vram_max_allocated_mb:.2f} MB")
    else:
        print("GPU VRAM Profiling:    N/A (CPU Mode)")
    print("==================================================\n")

if __name__ == "__main__":
    run_benchmark()
