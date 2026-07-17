# Contributing to PathVision Final

Thank you for your interest in contributing to PathVision Final! This document outlines our repository workflow, coding guidelines, and pull request checklist.

---

## 1. Code Quality & Style Guidelines

To keep the codebase maintainable, all contributions must follow these conventions:

- **Type Hints**: All functions and methods must include explicit type hints:
  ```python
  def build_mesh(self, mask: np.ndarray) -> NavigationMesh:
  ```
- **Forward Reference**: Declare `from __future__ import annotations` at the top of every python module.
- **Style Standard**: Adhere to PEP 8 guidelines. Use a standard code formatter (e.g. `black` or `yapf`) before submitting your changes.
- **Thread Safety**: Never write blocking visual code. Ensure locks are acquired with timeouts, and restrict PyTorch execution using `torch.set_num_threads()` to prevent core saturation.

---

## 2. Git Workflow & Branch Naming

We use a standard branch-and-PR model:

1. **Fork the Repository**: Create your own copy of the repository.
2. **Branch from main**: Create a feature branch with a descriptive name:
   - `feat/your-feature` for additions.
   - `fix/your-fix` for bug fixes.
   - `docs/your-doc` for updates to documentation.
3. **Commit Messages**: Follow standard conventional commits formatting (e.g., `feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`).

---

## 3. Pull Request Checklist

Before submitting a Pull Request, verify that your changes pass the following checks:

- [ ] Syntax compilation check executes without warnings:
  ```bash
  python -m py_compile your_modified_file.py
  ```
- [ ] No regression is introduced to the target hardware performance benchmarks ($<30\text{ ms}$ vision latency).
- [ ] No GGUF or ONNX weights files are accidentally included in your commit list.
- [ ] All code modifications are documented with docstrings and comments.
