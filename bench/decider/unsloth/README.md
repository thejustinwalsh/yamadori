# Unsloth decision-model training, and what it means for jjava (operator, 2026-10-07)

Background and the guide itself: docs/research/DECIDER-RESEARCH.md Part 5. Runtime: `.venv-unsloth` (gitignored by the
venv's own `.gitignore`), locked in `locks/unsloth.lock.txt` (unsloth 2026.10.2, torch 2.11.0+cu128, transformers 5.17.0).
Data and checkpoints live OUTSIDE git (`index/decider_train/`, `models/` below the models dir); only code and small result
files are committed.

Files are added phase by phase; this index is kept current.

| phase | what | files |
|---|---|---|
| 1 | knowledge base | docs/research/DECIDER-RESEARCH.md Part 5, docs/JJAVA.md section 11 |
| 2 | training and tests (venv: .venv-unsloth) | prepare_data.py (data), train_decision.py (recipe), lanes.py (pause/resume the job lanes), eval_jevbench.py, skills_prepare.py + eval_skills.py + skills_score.py (our sets), data_manifest.json; RESULT-q08-e1-run1.md (run 1: baseline only, weights lost) |
| 4 | frozen-Bonsai head | bonsai_head.py (records / extract / train / eval / selftest), DESIGN-bonsai-head.md |
