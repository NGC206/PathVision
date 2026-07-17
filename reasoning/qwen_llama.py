from __future__ import annotations

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Any

# Configure logging
LOGGER = logging.getLogger(__name__)

# Setup DLL search paths for Windows CUDA dynamic backends before importing llama_cpp
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
llama_bin_path = r"D:\Work\BDS\llama.cpp\build\bin\Release"

_dll_dirs = []
if os.name == "nt":
    if os.path.exists(cuda_path):
        try:
            _dll_dirs.append(os.add_dll_directory(cuda_path))
            LOGGER.info("Added CUDA path to DLL search directories: %s", cuda_path)
        except Exception as e:
            LOGGER.warning("Could not add CUDA DLL directory: %s", e)
    if os.path.exists(llama_bin_path):
        try:
            _dll_dirs.append(os.add_dll_directory(llama_bin_path))
            LOGGER.info("Added llama.cpp Release bin path to DLL search directories: %s", llama_bin_path)
        except Exception as e:
            LOGGER.warning("Could not add llama.cpp DLL directory: %s", e)
            
    # Prepend to PATH environment variable to assist nested dynamic LoadLibrary calls in ggml
    paths = [p for p in [llama_bin_path, cuda_path] if os.path.exists(p)]
    if paths:
        os.environ["PATH"] = ";".join(paths) + ";" + os.environ["PATH"]
        os.environ["LLAMA_CPP_LIB"] = os.path.join(llama_bin_path, "llama.dll")
        os.environ["GGML_BACKEND_PATH"] = os.path.join(llama_bin_path, "ggml-cuda.dll")

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    LOGGER.warning("Failed to import llama-cpp-python: %s", e)
    LLAMA_CPP_AVAILABLE = False

from reasoning.prompts import build_qwen_messages, fallback_instruction

class QwenLlamaReasoner:
    """Native llama.cpp adapter for Qwen-2.5-VL-3B-Instruct inference."""

    def __init__(self, config: Any) -> None:
        self._cfg = config.reasoning
        self._enabled = self._cfg.enabled
        self.model = None
        self.backend_type = "CPU"
        self._lock = threading.Lock()
        
        # Timing statistics
        self.load_time_seconds = 0.0
        
        if not self._enabled:
            LOGGER.info("[LLAMA.CPP] Reasoner disabled in configuration.")
            return

        if not LLAMA_CPP_AVAILABLE:
            LOGGER.error("[LLAMA.CPP] llama-cpp-python package is not available. Falling back to deterministic instructions.")
            return

        model_path = Path(self._cfg.model_path)
        llama_cpp_root = Path(self._cfg.llama_cpp_root)

        # Path verification
        if not llama_cpp_root.exists():
            LOGGER.error(
                "\n[LLAMA.CPP] Missing llama.cpp root path!\n"
                "Path: %s\n"
                "Suggested fix: Please ensure the compiled llama.cpp repository is located at the configured path.\n",
                llama_cpp_root
            )
            return

        if not model_path.exists():
            LOGGER.error(
                "\n[LLAMA.CPP] Missing GGUF model file!\n"
                "Path: %s\n"
                "Suggested fix: Download the Qwen2.5-VL-3B-Instruct model and place it in the models/ directory.\n",
                model_path
            )
            return

        # Load GGUF model exactly once
        t0 = time.perf_counter()
        try:
            LOGGER.info("[LLAMA.CPP] Loading model from %s...", model_path)
            self.model = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=self._cfg.context_length,
                n_gpu_layers=self._cfg.gpu_layers,
                n_threads=self._cfg.threads,
                verbose=False
            )
            self.load_time_seconds = time.perf_counter() - t0
            
            # Detect backend type
            # If any layers are offloaded to GPU, it uses the dynamic backend
            if self._cfg.gpu_layers != 0:
                self.backend_type = "CUDA"
            else:
                self.backend_type = "CPU"
                
            LOGGER.info(
                "[LLAMA.CPP] Model loaded successfully in %.2f seconds.\n"
                "Backend: %s | GPU Layers: %d | Context: %d",
                self.load_time_seconds, self.backend_type, self._cfg.gpu_layers, self._cfg.context_length
            )
        except Exception as e:
            LOGGER.exception(
                "[LLAMA.CPP] Failed to load native llama.cpp model: %s\n"
                "Falling back to deterministic guidance.", e
            )
            self.model = None

    def generate(self, scene_payload: dict[str, Any], recent_memories: list[str], mode_str: str) -> str:
        """Thread-safe generation of companion guidance using native llama.cpp."""
        if not self._enabled or self.model is None:
            return fallback_instruction(scene_payload, mode_str)

        # Build messages using existing prompts.py helper
        messages = build_qwen_messages(scene_payload, recent_memories, mode_str)
        
        acquired = self._lock.acquire(timeout=0.5)
        if not acquired:
            LOGGER.warning("[LLAMA.CPP] Reasoner lock busy. Returning fallback.")
            return fallback_instruction(scene_payload, mode_str)
        try:
            try:
                t0 = time.perf_counter()
                response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    top_p=self._cfg.top_p,
                    repeat_penalty=self._cfg.repeat_penalty
                )
                dt = time.perf_counter() - t0
                text = str(response["choices"][0]["message"]["content"]).strip()
                if text.upper() == "SILENT":
                    return ""
                
                # Compute tokens per second
                prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
                total_tokens = prompt_tokens + completion_tokens
                tps = completion_tokens / max(dt, 1e-6)
                
                LOGGER.debug(
                    "[LLAMA.CPP] Generated: '%s' | Tokens: %d prompt + %d gen | Speed: %.1f t/s | Latency: %.2fs",
                    text, prompt_tokens, completion_tokens, tps, dt
                )
                return text or fallback_instruction(scene_payload, mode_str)
            except Exception as e:
                LOGGER.error("[LLAMA.CPP] Chat completion failed: %s. Using fallback.", e)
                return fallback_instruction(scene_payload, mode_str)
        finally:
            self._lock.release()

    def close(self) -> None:
        """Release llama.cpp resources."""
        LOGGER.info("[LLAMA.CPP] Closing reasoner backend...")
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
        LOGGER.info("[LLAMA.CPP] Reasoner backend closed.")
