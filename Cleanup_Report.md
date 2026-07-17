# Cleanup Report — PathVision Final

This report classifies all files in the repository into cleanup categories. In accordance with Safe Mode instructions, no files will be automatically deleted; they are marked here for manual archive, removal, or gitignore exclusion.

---

## 1. Categorized Repository Files

### A. SAFE TO KEEP (Source Code & Core Configurations)
- `main.py` — Orchestrates runtime thread loops.
- `config.py` — Holds dataclass settings and defaults.
- `requirements.txt` — Logically grouped dependency list.
- `LICENSE` — Project MIT License text.
- `perception/*.py` — Vision pipelines, TRT engines, and scene fusion.
- `navigation/*.py` — Spatial steering and geometry.
- `reasoning/*.py` — Prompt and situations.
- `speech/*.py` — Kokoro TTS.
- `learning/*.py` — Scene loggers and feedback.
- `utils/trt_utils.py` — Shared CUDA utility.
- `tests/benchmark.py`, `tests/stress_test.py` — Verification suites.

### B. SAFE TO ARCHIVE (Move to `report/` or `research/`)
- `final_project_report.doc` (Move to `/report/`) — Unified MS Word report.
- `final_project_report.md` (Move to `/report/`) — Duplicate markdown handbook content.
- `final_project_report_v1.doc` (Move to `/report/`) — Previous Word draft.
- `simple_project_report.md` (Move to `/report/`) — Smaller markdown summary.
- `validation_report.md` (Move to `/report/`) — Previous validation report.

### C. GENERATED OUTPUTS (Exclude via Gitignore)
- `output/validation/results/` — Batch validation output files (depths, masks, overlays).
- `logs/` — Runtime telemetry logs (`*.log`, `*.jsonl`).
- `learning/collected_data/` — Frames captured during confidence drops.

### E. CACHE (Exclude via Gitignore)
- `**/__pycache__/` — Python bytecode cache.
- `**/*.py[cod]` — Compiled python files.
- `**/*.pyd` — Compiled native extensions.

### F. MACHINE-SPECIFIC LARGE BINARIES (Exclude via Gitignore & Document in README)
- `models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` — 1.93GB LLM.
- `engines/depth_vits_fp16.engine` — Compiled Depth Anything.
- `engines/pathvision.engine` — Compiled PathVision.
- `engines/custom.engine` — Test engine.
