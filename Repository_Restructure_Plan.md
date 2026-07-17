# Repository Restructure Plan — PathVision Final

This document maps the proposed transition of files from the current folder structure to the restructured layout optimized for a professional GitHub release.

---

## 1. Mapping Current to Proposed Structure

```
Current Path                                Proposed Path                              Justification
========================================================================================================================
[Root Files]
/final_project_report.doc                  /report/final_project_report.doc           Isolate compiled Word document.
/final_project_report_v1.doc               /report/final_project_report_v1.doc        Archive old compiled report.
/final_project_report.md                   /report/final_project_report.md            Archive duplicate markdown report.
/simple_project_report.md                  /report/simple_project_report.md           Archive simplified report.
/validation_report.md                      /report/validation_report.md               Archive old validation stats.
/output_cli.txt                            [DELETED/IGNORED]                          Obsolete debug terminal output.

[Documentation files under docs/]
/docs/Architecture.md                      /docs/ARCHITECTURE.md                      Standardize case to uppercase.
/docs/Future.md                            /docs/FUTURE.md                            Standardize case to uppercase.
/docs/Installation.md                      /docs/INSTALLATION.md                      Standardize case to uppercase.
/docs/Performance.md                       /docs/PERFORMANCE.md                       Standardize case to uppercase.
/docs/Pipeline.md                          /docs/PIPELINE.md                          Standardize case to uppercase.

[Assets & Visual Figures]
/report_results/hallway_blocked_report.png /assets/validation/00002_report.png        Move validation overlay comparison.
/report_results/hallway_clear_report.png   /assets/validation/00001_report.png        Move validation clear comparison.
/report_results/noise_texture_report.png   /assets/validation/00003_report.png        Move noise validation comparison.

[New Markdown Specifications]
(none)                                     /models/README.md                          Explain model weights downloads.
(none)                                     /engines/README.md                         Explain TensorRT compilation.
(none)                                     /CODE_OF_CONDUCT.md                        Add contributor code of conduct.
(none)                                     /SECURITY.md                               Add vulnerability reporting guide.
(none)                                     /AI_DEVELOPMENT.md                         Document AI assistance transparency.
(none)                                     /VERSION                                   Store project release version.
(none)                                     /RELEASE_NOTES.md                          Store project release notes.
========================================================================================================================
```

---

## 2. Restructured Repository Layout Diagram

After applying these changes, the clean repository tree will appear as follows:

```
PathVision_Final/
├── README.md                           # Professional root landing page
├── LICENSE                             # MIT License file
├── CHANGELOG.md                        # Version changelog
├── CONTRIBUTING.md                     # Pull request conventions
├── CODE_OF_CONDUCT.md                  # Contributor Code of Conduct
├── SECURITY.md                         # Security policy
├── AI_DEVELOPMENT.md                   # AI transparency disclosure
├── VERSION                             # File containing version "0.1.0-alpha"
├── RELEASE_NOTES.md                    # Detailed release notes
├── requirements.txt                    # Logically grouped dependencies
├── .gitignore                          # Exhaustive open-source ignore list
├── main.py                             # Project main orchestrator
├── config.py                           # Project configuration
├── perception/                         # Vision adapters
├── navigation/                         # Geometric decision code
├── reasoning/                          # LLM handlers
├── speech/                             # TTS audio code
├── learning/                           # Logging and auto-label modules
├── utils/                              # Shared trt utilities
├── tests/                              # Verification test suites
├── research/                           # Academic scripts, loss, training
├── docs/                               # All uppercase technical documents
│   ├── ARCHITECTURE.md
│   ├── RUNTIME.md
│   ├── NAVIGATION.md
│   ├── AI_MODELS.md
│   ├── TRAINING.md
│   ├── DATASET.md
│   ├── PERFORMANCE.md
│   ├── BENCHMARKS.md
│   ├── FILE_STRUCTURE.md
│   ├── DEVELOPMENT.md
│   ├── API_REFERENCE.md
│   ├── TROUBLESHOOTING.md
│   ├── REPORT.md
│   ├── ROADMAP.md
│   ├── INSTALLATION.md
│   ├── FUTURE.md
│   └── PIPELINE.md
├── assets/                             # Image assets and diagrams
│   ├── architecture/
│   ├── runtime/
│   ├── validation/
│   └── screenshots/
├── report/                             # Compiled Word and PDF handbooks
├── models/
│   └── README.md                       # Model downloader guide
└── engines/
    └── README.md                       # trtexec compiler guide
```
