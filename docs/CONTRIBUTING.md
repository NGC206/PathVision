# Contributing Guidelines — PathVision Final

This document outlines the coding standards, repository workflow, pull request processes, and formatting rules for contributing to the PathVision Final project.

---

## 1. Code Quality & Formatting Standards

To maintain a clean and maintainable codebase, all contributions must adhere to the following standards:

### A. Python Coding Standards
- Follow **PEP 8** style guidelines.
- Use explicit type hints for all function arguments and return values:
  ```python
  def process_mask(self, mask: np.ndarray, threshold: float) -> np.ndarray:
  ```
- Declare `from __future__ import annotations` at the top of every file to enable forward type declarations.
- Use uppercase for global configuration constants (e.g. `MODEL_W`, `SAFE_CLASS_ID`).

### B. Thread-Safety Policies
- Never instantiate blocking I/O calls inside the visual `FastVisionPipeline` loop.
- Use reentrant locks (`threading.RLock`) when reading or writing variables shared across threads.
- Limit external PyTorch thread usage: when initializing Torch-based models, explicitly restrict thread consumption using `torch.set_num_threads()` to prevent logical core saturation.

---

## 2. Repository Git Workflow

We use a feature branch git workflow:

1. **Clone & Fork**: Create a feature branch from the `main` branch.
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. **Commit Conventions**:
   Commit messages should be clear and follow the conventional commit structure:
   - `feat: ...` for new features or capabilities.
   - `fix: ...` for bug fixes or stabilization updates.
   - `docs: ...` for documentation additions.
   - `perf: ...` for latency optimizations.
3. **Local Testing**:
   Before committing, run compiler checks and stress tests locally:
   ```powershell
   python -m py_compile your_modified_file.py
   python tests/stress_test.py
   ```
4. **Pull Requests**:
   Open a pull request to the `main` branch. Ensure that:
   - Your branch builds cleanly without compilation errors.
   - You provide a summary of modifications, tested hardware, and resource stats.
   - The code passes review by at least one core engineer.
