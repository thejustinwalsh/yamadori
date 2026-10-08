# Unsloth decision-model training, and what it means for jjava (operator, 2026-10-07)

Background and the guide itself: docs/research/DECIDER-RESEARCH.md Part 5. Runtime: `.venv-unsloth` (gitignored by the
venv's own `.gitignore`), locked in `locks/unsloth.lock.txt` (unsloth 2026.10.2, torch 2.11.0+cu128, transformers 5.17.0).
Data and checkpoints live OUTSIDE git (`index/decider_train/`, `models/` below the models dir); only code and small result
files are committed.

Files are added phase by phase; this index is kept current.

| phase | what | files |
|---|---|---|
| 1 | knowledge base | docs/research/DECIDER-RESEARCH.md Part 5, docs/JJAVA.md section 11 |
