# Pruning-Induced Decision-Boundary Reallocation in Fine-Grained IoT Intrusion Classification

**Repository companion for the Computer Networks submission.**

Author: Md Anas Biswas, School of Computing, University of Portsmouth.

## What the archived experiments support

The primary experiment studies a 34-class CICIoT2023 CNN1D under M0, prune50, prune80, distillation, float16, and dynamic int8 quantisation of `Linear` layers.

The bounded findings are:

- Moderate pruning can create class-specific recall losses that are not localised by aggregate scores.
- Under prune80, the main attack errors are predominantly **attack-to-attack fine-type substitutions**, not demonstrated silent attack-to-benign passage. Benign traffic is also routed to attack labels, creating a separate false-alert burden.
- Under the archived identical-split probe and the stricter train/validation-to-test replication, recall changes much more than relative linear decodability. The conclusion is retained decodability, not perfect representational invariance.
- Affected samples consolidate onto dominant or decision-favoured confusable labels. Raw class frequency does not consistently determine the transfer direction.
- Body-only pruning with a dense trainable head reproduces the persistent class-failure pattern more closely than head-only pruning. Classification-head deletion is therefore not a complete explanation.
- A dense replacement head recovers aggregate macro-F1 but reduces whole-model sparsity and transfers error to some former absorber classes. It is a recall-recovery trade-off, not a cost-free trust-preserving method.
- The archived baseline diagnostic is **retrospective** because its covariates were assembled on the evaluation partition. It is not presented as validated pre-deployment forecasting.
- TON_IoT provides a qualitative external collapse-probe-recovery check, not population-wide generality.

The current CNN int8 cell is **Linear-layer dynamic quantisation**, leaving the Conv1d body in FP32. It is not full-model CNN int8.

## Computer Networks extension notebooks

The extension notebooks write only new outputs under `results/tables/comnet/`.

| Notebook | Purpose | Compute |
|---|---|---|
| `09_comnet_security_semantics_and_recovery.ipynb` | Binary, alert-family, fine-type, attack-to-benign, benign-to-attack, substitution, calibration, and recovered-head audit | Light/medium if checkpoints exist |
| `10_validation_tiers_and_safe_prediction.ipynb` | Validation-frozen tiers and validation-only predictor covariates with test-only targets and family-held-out evaluation | Light/medium if checkpoints or archived targets exist |
| `11_independent_paired_pruning_seeds.ipynb` | Independently train five baselines and prune each at 50/80%; paired effects and mask overlap | Heavy GPU run |
| `12_strict_probes_batchnorm_sparse_head.ipynb` | Strict train/validation-to-test probes, strict pairwise probes, BatchNorm-only recalibration, dense-versus-sparse replacement-head control | Medium/heavy |
| `13_comnet_deployment_benchmark.ipynb` | Actual serialized bytes, visible nonzeros, dense CPU latency, p95 latency, throughput, and RSS snapshots | CPU; run on reportable hardware |

Read `COMPUTER_NETWORKS_RUN_ORDER.md` before running them.

## Canonical archived code

| Manuscript component | Canonical code |
|---|---|
| Baseline training/evaluation | `src/train.py` |
| Compression matrix | `src/compression.py` |
| Archived relative probes, consolidation, and rank | `src/explain.py` |
| Archived retrospective diagnostic | `src/predict.py` |
| Decision-layer recovery | `src/mitigate.py` |
| Computer Networks extension analyses | `src/comnet_audit.py`, notebooks 09-13 |

`src/crux.py` and `src/diagnostic.py` are tested compatibility interfaces. They replace former dead public stubs but do not retroactively generate archived results.

## Safe setup

Never put GitHub, Kaggle, cloud, or API credentials in source files or notebooks.

```bash
git clone https://github.com/anasbiswas1/iot-trust-compression.git
cd iot-trust-compression
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-comnet.txt
```

For Colab, mount Drive and use the repository already stored there. Do not paste a token into a notebook.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO = '/content/drive/MyDrive/IoT_Trust_Research/iot-trust-compression'
os.chdir(REPO)
sys.path.insert(0, REPO)
```

## Result boundaries

- Existing files under `results/` are archived evidence for the manuscript.
- The update package does not overwrite archived results, datasets, models, or logs.
- New extension outputs are written under `results/tables/comnet/`.
- Do not update manuscript numbers until the new CSVs have been inspected and independently checked.

## Release discipline

Before journal submission:

1. complete the required extension notebooks;
2. reconcile every inserted number against its CSV;
3. remove any credential file from the working tree and Git history;
4. run the repository tests;
5. create a tagged release;
6. record the exact tag and commit hash in the manuscript.

See `REPRODUCIBILITY.md`, `SECURITY.md`, `COMPUTER_NETWORKS_RUN_ORDER.md`, and `MANUSCRIPT_UPDATE_MAP.md`.
