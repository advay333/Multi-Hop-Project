# Multi-Hop Table Reasoning via RL

A small learned policy chooses filters and joins over a relational database, SQLite executes
them, and a frozen LLM reader answers from the resulting sub-table. The policy is trained with
behaviour cloning, then PPO / GRPO, using potential-based rewards derived from gold SQL.

| Document | What it is |
|---|---|
| `Multi-Hop_Table_RL_Project_Report.pdf` | Problem, literature, MDP, rewards, metrics, algorithm arms |
| `DESIGN.md` | Architecture: modules, interfaces, data model, storage |
| `PLAN.md` | Build order (phases P0–P10) with a gate per phase |
| `data/DATASET_CARD.md` | Data provenance and gate numbers |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
```

All data lives in private HuggingFace repos (`DESIGN.md` §4); set `HF_TOKEN` to access them.
Raw sources are mirrored by `notebooks/source_data.py` (molab); nothing large is stored in git.
