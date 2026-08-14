What's on GitHub now: PROJECT.md, TASK_01_DATASET.md, the three reference PDFs, requirements.txt, .gitignore, data/DATASET_CARD.md, and src/mmqa_schema_probe.py + src/mmqa_ingest.py.

Deliberately excluded (via .gitignore): .venv/ and everything under data/raw|db|cache|trajectories|splits/ — that's ~2.8GB of downloaded/generated data, including one 879MB file that GitHub's 100MB-per-file limit would hard-reject outright. DATASET_CARD.md's "Regeneration" section already has the exact commands to rebuild all of it locally, so nothing is lost — anyone cloning the repo just re-runs Phase 0.
