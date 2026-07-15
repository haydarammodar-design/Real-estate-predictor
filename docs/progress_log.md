# Progress & Change Log

> Use this file to track all changes made after the initial report version.
> Each update should be appended as a new entry with date, author, description,
> and affected files.

---

## Log Format

```markdown
### YYYY-MM-DD — [Brief Title]

**Author:** [Name]
**Affected files:**
- `path/to/file.py`
- `path/to/template.html`

**Changes:**
1. What was changed
2. Why it was changed
3. Any metrics or results affected

**Metrics update (if applicable):**
| Metric | Old | New |
|--------|:---:|:---:|
| MAE | X | Y |
| R² | X | Y |

**Validation:** [How the change was tested]
**Status:** [Done / In Progress / Rolled Back]
```

---

## Entries

<!-- New entries go below this line, most recent first -->

### 2026-07-15 — Final Delivery Hardening

**Author:** Haidra Mohammad

**Affected files:**
- `src/data_processor.py`
- `src/dvf_downloader.py`
- `src/modeling.py`
- `src/train.py`
- `src/train_apartment.py`
- `src/train_land.py`
- `src/app.py`
- `src/templates/index.html`
- `tests/test_core.py`
- `README.md`
- `docs/final_delivery_checklist.md`
- `docs/technical_appendix.md`
- `docs/data_sources.md`

**Changes:**
1. Replaced random-split comparable features with a 2024-to-2025 point-in-time validation protocol.
2. Corrected DVF mutation handling for future source rebuilds and added timeout-controlled parallel downloads.
3. Added residual-land-reference disclosure, CES/floor-area-ratio feasibility calculations, and integrated development margin/ROI outputs.
4. Corrected investment IRR/ROI to include purchase and sale costs.
5. Added input validation, external-service request limits, writable deployment caches, tests, dependency lock, release manifest generator, and handoff documentation.

**Validation:** Corrected national DVF release snapshot built (4,006,005 rows; 93 departments/year; 2021-2025), all three models trained with a 2024-to-2025 point-in-time protocol, release manifest generated, compilation and 4 unit tests pass, and API acceptance checks pass.
**Status:** Done

### 2026-06-12 — Initial Report

**Author:** System
**Affected files:**
- `docs/internship_report.md` (created)
- `docs/technical_appendix.md` (created)
- `docs/progress_log.md` (created)

**Changes:** Created initial documentation for the AlfaScript Real Estate Predictor project.
