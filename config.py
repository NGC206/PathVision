"""Central runtime configuration for PathVision Final."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnginePaths:
    """Filesystem paths to TensorRT engine artifacts."""

    pathvision: Path
    depth_anything: Path


@dataclass(frozen=True)
class CameraSettings:
    """Camera source and acquisition settings."""

    index: int
    width: int
    height: int
    backend_dshow: bool


@dataclass(frozen=True)
class PathVisionSettings:
    """PathVision segmentation settings."""

    model_width: int
    model_height: int
    safe_class_id: int
    safe_probability_threshold: float
    display_scale: float


@dataclass(frozen=True)
class DepthSettings:
    """Depth Anything runtime settings."""

    input_width: int
    input_height: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    nearest_obstacle_quantile: float


@dataclass(frozen=True)
class NavigationSettings:
    """Safety and decision thresholds."""

    deadband_ratio: float
    min_safe_area_ratio: float
    min_bottom_width_ratio: float
    minimum_clearance: float
    caution_clearance: float


@dataclass(frozen=True)
class ReasoningSettings:
    """Qwen runtime settings."""

    enabled: bool
    backend: str
    model_path: Path
    llama_cpp_root: Path
    context_length: int
    gpu_layers: int
    threads: int
    max_tokens: int
    temperature: float
    top_p: float
    repeat_penalty: float
    warmup: bool
    update_interval_seconds: float
    generation_timeout_seconds: float


@dataclass(frozen=True)
class SpeechSettings:
    """Kokoro speech synthesis settings."""

    enabled: bool
    voice: str
    language_code: str
    speed: float
    sample_rate: int
    cooldown_seconds: float


@dataclass(frozen=True)
class LearningSettings:
    """Offline learning/logging settings."""

    enabled: bool
    scene_log_path: Path
    feedback_log_path: Path
    dataset_output_dir: Path
    capture_confidence_threshold: float


@dataclass(frozen=True)
class RuntimeSettings:
    """General runtime behavior and diagnostics settings."""

    show_preview: bool
    save_preview_frames: bool
    log_level: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration object."""

    project_root: Path
    output_dir: Path
    engines: EnginePaths
    camera: CameraSettings
    pathvision: PathVisionSettings
    depth: DepthSettings
    navigation: NavigationSettings
    reasoning: ReasoningSettings
    speech: SpeechSettings
    learning: LearningSettings
    runtime: RuntimeSettings


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    """Load application configuration from defaults and environment variables."""

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "output"
    logs_dir = project_root / "logs"
    learning_dir = project_root / "learning"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    learning_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        project_root=project_root,
        output_dir=output_dir,
        engines=EnginePaths(
            pathvision=Path(
                os.getenv("PATHVISION_ENGINE_PATH", str(project_root / "engines" / "pathvision.engine"))
            ),
            depth_anything=Path(
                os.getenv("DEPTH_ENGINE_PATH", str(project_root / "engines" / "depth_vits_fp16.engine"))
            ),
        ),
        camera=CameraSettings(
            index=_env_int("CAMERA_INDEX", 0),
            width=_env_int("CAMERA_WIDTH", 640),
            height=_env_int("CAMERA_HEIGHT", 480),
            backend_dshow=_env_bool("CAMERA_BACKEND_DSHOW", True),
        ),
        pathvision=PathVisionSettings(
            model_width=_env_int("PATHVISION_MODEL_WIDTH", 320),
            model_height=_env_int("PATHVISION_MODEL_HEIGHT", 240),
            safe_class_id=_env_int("PATHVISION_SAFE_CLASS_ID", 1),
            safe_probability_threshold=_env_float("PATHVISION_SAFE_PROB_THRESHOLD", 0.55),
            display_scale=_env_float("DISPLAY_SCALE", 2.0),
        ),
        depth=DepthSettings(
            input_width=_env_int("DEPTH_INPUT_WIDTH", 518),
            input_height=_env_int("DEPTH_INPUT_HEIGHT", 518),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            nearest_obstacle_quantile=_env_float("DEPTH_NEAREST_QUANTILE", 0.10),
        ),
        navigation=NavigationSettings(
            deadband_ratio=_env_float("NAV_DEADBAND_RATIO", 0.08),
            min_safe_area_ratio=_env_float("NAV_MIN_SAFE_AREA_RATIO", 0.04),
            min_bottom_width_ratio=_env_float("NAV_MIN_BOTTOM_WIDTH_RATIO", 0.10),
            minimum_clearance=_env_float("NAV_MINIMUM_CLEARANCE", 0.05),
            caution_clearance=_env_float("NAV_CAUTION_CLEARANCE", 0.15),
        ),
        reasoning=ReasoningSettings(
            enabled=_env_bool("QWEN_ENABLED", True),
            backend=os.getenv("QWEN_BACKEND", "llama.cpp"),
            model_path=Path(os.getenv("LLAMA_MODEL_PATH", str(project_root / "models" / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"))),
            llama_cpp_root=Path(os.getenv("LLAMA_CPP_ROOT", r"D:\Work\BDS\llama.cpp")),
            context_length=_env_int("LLAMA_CONTEXT", 2048),
            gpu_layers=_env_int("LLAMA_GPU_LAYERS", 16),
            threads=_env_int("LLAMA_THREADS", 4),
            max_tokens=_env_int("LLAMA_MAX_TOKENS", 60),
            temperature=_env_float("LLAMA_TEMPERATURE", 0.2),
            top_p=_env_float("LLAMA_TOP_P", 0.9),
            repeat_penalty=_env_float("LLAMA_REPEAT_PENALTY", 1.1),
            warmup=_env_bool("QWEN_WARMUP", True),
            update_interval_seconds=_env_float("QWEN_UPDATE_INTERVAL_SECONDS", 1.0),
            generation_timeout_seconds=_env_float("QWEN_GENERATION_TIMEOUT_SECONDS", 4.0),
        ),
        speech=SpeechSettings(
            enabled=_env_bool("KOKORO_ENABLED", True),
            voice=os.getenv("KOKORO_VOICE", "af_heart"),
            language_code=os.getenv("KOKORO_LANG_CODE", "a"),
            speed=_env_float("KOKORO_SPEED", 1.0),
            sample_rate=_env_int("KOKORO_SAMPLE_RATE", 24000),
            cooldown_seconds=_env_float("KOKORO_COOLDOWN_SECONDS", 1.5),
        ),
        learning=LearningSettings(
            enabled=_env_bool("LEARNING_ENABLED", True),
            scene_log_path=Path(os.getenv("SCENE_LOG_PATH", str(logs_dir / "scene_log.jsonl"))),
            feedback_log_path=Path(os.getenv("FEEDBACK_LOG_PATH", str(logs_dir / "feedback.jsonl"))),
            dataset_output_dir=Path(os.getenv("DATASET_OUTPUT_DIR", str(learning_dir / "collected_data"))),
            capture_confidence_threshold=_env_float("CAPTURE_CONFIDENCE_THRESHOLD", 0.45),
        ),
        runtime=RuntimeSettings(
            show_preview=_env_bool("SHOW_PREVIEW", True),
            save_preview_frames=_env_bool("SAVE_PREVIEW_FRAMES", False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        ),
    )