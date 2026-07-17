# GitHub Release Checklist — PathVision Final

Use this checklist to verify repository readiness before making the first public commit or publishing a release tag.

---

## Pre-Release Verification Checklist

- [ ] **Repository Identity**:
  - Name set to `PathVision_Final`.
  - Topics set (e.g. `tensorrt`, `computer-vision`, `accessibility`, `local-llm`, `tts`, `real-time`).
- [ ] **Release Metadata**:
  - `VERSION` file contains `0.1.0-alpha`.
  - `RELEASE_NOTES.md` is complete and lists current capabilities and limitations.
- [ ] **Large Checkpoint Exclusion**:
  - Verified no `.engine` files exist in git tracking list.
  - Verified no `.onnx` or `.pth` weights exist in git tracking list.
  - Verified no `.gguf` file exists in git tracking list.
- [ ] **Folder Structure & Reorganization**:
  - Move report files (`*.doc`, duplicate markdown reports) into `/report/`.
  - Reorganize documentation files in `/docs/` and standardize filenames to uppercase.
  - Create the `assets/` subdirectories (`architecture/`, `runtime/`, etc.).
- [ ] **Configuration Files**:
  - `.gitignore` verified to block caching, logs, virtual environments, and intermediate validation outputs.
  - `requirements.txt` cleaned, duplicates removed, and organized by group.
- [ ] **Documentation Completeness**:
  - Root `README.md` complete and references `PathVision_Final_Documentation.docx` as the master documentation.
  - `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `AUTHORS.md` created.
  - `AI_DEVELOPMENT.md` transparency statement added.
