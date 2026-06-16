# iot-trust-compression

**Measure → explain → predict → mitigate: per-class trustworthiness collapse in compressed network IoT intrusion detectors.**

Author: Md Anas Biswas, School of Computing, University of Portsmouth (up2082724@myport.ac.uk). Sole-authored.
Target: Q1 — Expert Systems with Applications (ESWA) or Computers & Security.

The headline contribution is **PREDICT**: a pre-deployment, per-class diagnostic that forecasts which attack classes lose recall / calibration / explanation-trust under compression, computed from the baseline model *before* compressing — and grounded in a capacity-allocation theory (Minority Collapse), not a correlation. See `PLAN.md` for the full design and `logs/anchor_preregistration.json` for the frozen commitments.

---

## First-time setup (run once)

Open **`SETUP.ipynb`** in a fresh Colab session and run it top to bottom. It mounts Drive, unzips this bundle to `MyDrive/IoT_Trust_Research/iot-trust-compression/`, configures git, connects an **empty** GitHub repo (`anasbiswas1/iot-trust-compression`) via a `ghp_...` token, sanity-checks that `data/` and `models/` are ignored, and makes the first commit. Then delete `SETUP.ipynb` — you won't need it again. After that, the per-notebook end-of-unit ritual (`git add -A && git commit && git push`) is the whole workflow.

## Repository layout

```
config/        config.yaml       # SINGLE SOURCE OF TRUTH (paths, seeds, splits, matrix, archs)
src/           config.py         # loads config.yaml, resolves ALL paths (never hardcode a path)
               data.py           # load, dedup + identity removal (mechanism-critical), grouped split
               models.py         # MLP, CNN1D, FT-Transformer (+GRU fallback); all expose .features()
               compression.py    # M0/prune50/prune80/distillation/int8/float16 (PyTorch-native)
               metrics.py        # per-class recall, ECE, stability vs faithfulness, bootstrap CIs
               geometry.py       # neural-collapse geometry, effective rank, margin (v2.1 theory layer)
               crux.py           # linear-probe recoverability (the spine)
               diagnostic.py     # the headline predictor + baselines-to-beat + generalisation eval
               mitigate.py       # predictor-guided protection + conformal certificate (stretch)
notebooks/     00..08            # one stage per notebook, self-contained (see below)
results/       tables/ figures/ arrays/   # tracked in git — these back the paper
logs/          anchor_preregistration.json # frozen BEFORE any trust metric
requirements.txt   PLAN.md   README.md
```

Data and model binaries are **gitignored** and live only in Drive. GitHub holds code, notebooks, result tables, figures, and logs.

## Drive + GitHub (the working pattern)

- GitHub repo: `github.com/anasbiswas1/iot-trust-compression`
- Drive root: `/content/drive/MyDrive/IoT_Trust_Research/iot-trust-compression/` — clone the repo **inside** this folder so Colab reads/writes one tree.

## Colab bootstrap (top of every notebook)

```python
from google.colab import drive; drive.mount('/content/drive')
import os, sys
REPO = '/content/drive/MyDrive/IoT_Trust_Research/iot-trust-compression'
os.chdir(REPO); sys.path.insert(0, REPO)
from src.config import CFG, PATHS, set_all_seeds, require_frozen
set_all_seeds(CFG['anchor_seed'])
```

## The one rule that prevents the last disaster

Every path comes from `PATHS` (which reads `config/config.yaml`). **No notebook or module ever builds a path inline** with `Path(REPO)/'...'`. The X-IDS multi-seed failure happened because downstream notebooks constructed output paths inline, so a seed/compression override couldn't redirect them — and the seed-42 baseline was overwritten. Centralised paths make seeded, locked splits actually hold. If you need a new path, add a method to `src/config.py`.

## Notebook order (one stage each, self-contained)

| nb | stage | purpose | gate / branch |
|----|-------|---------|---------------|
| `00_data_prep_and_split_audit` | 0 | load, dedup, identity removal, **group-count audit**, lock grouped splits | resolves grouping variable + 34-vs-8 granularity |
| `01_train_baselines` | 1 | train M0 anchors (CNN1D first), log params + macro-F1 | resolves remaining freeze-checklist items |
| `02_compress` | 1 | apply the compression matrix; int8 with transformer fallback | — |
| `03_measure_trust` | 1 | per-class recall / ECE / stability+faithfulness, seed null bands | **Stage 1 gate**: must replicate outside the null |
| `04_crux_probe` | crux | linear-probe recoverability + margin | **branch point**: info-loss vs decision-layer |
| `05_explain_mechanism` | 2 | hypothesis tournament, neural-collapse geometry, per-family causes | — |
| `06_predict_diagnostic` | 3 | the headline; beat freq + Tran&Fioretto; generalisation tests | across-arch test EARNS the causal claim |
| `07_mitigate` | 4 | predictor-guided protection; conformal certificate (stretch) | droppable per D3 |
| `08_robustness` | 5 | Bot-IoT (inverted), temporal-split drift check | only after 1–4 |

## End-of-unit discipline (non-negotiable)

After each notebook: (1) save outputs to Drive, (2) commit + push to GitHub with a meaningful message, (3) push the result CSVs / figures, (4) confirm Drive and GitHub are in sync. **Nothing uncommitted overnight.**

## Status

Scaffolding. The prereg is **pre-freeze** — set the freeze-checklist items in `config/config.yaml` (grouping variable, label granularity, param counts, matched macro-F1, crux threshold) from the first builds, then set `frozen_on` before computing any trust metric.
