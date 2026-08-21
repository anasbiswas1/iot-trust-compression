# Computer Networks repository update notes

## Files replaced or added

- bounded, Computer Networks-aligned `README.md`;
- hardened `.gitignore`;
- `SECURITY.md`, `REPRODUCIBILITY.md`, `COMPUTER_NETWORKS_RUN_ORDER.md`, and `MANUSCRIPT_UPDATE_MAP.md`;
- `requirements-comnet.txt`;
- tested compatibility modules `src/crux.py` and `src/diagnostic.py`;
- new `src/comnet_audit.py`;
- five extension notebooks, numbered 09-13;
- synthetic tests for compatibility and extension helpers.

## Safe text/metadata patches

The installer applies guarded, backward-compatible patches to:

- `src/mitigate.py`: rename the benign false-positive change while retaining old aliases;
- `config/config.yaml`: record the actual dynamic Linear-layer int8 method;
- `src/train.py`: replace outcome-seeking class-weighting rhetoric with neutral sensitivity wording;
- `src/predict.py`: label the default test-partition feature path as archived/retrospective and direct new pre-deployment work to validation.

## Files deliberately untouched

- `results/` archived numerical evidence;
- `data/`;
- `models/`;
- `logs/`;
- user credentials.

The optional `--remove-root-kaggle` flag removes only the working-tree `kaggle.json` after backing it up. It cannot clean Git history or revoke a credential; follow `SECURITY.md` separately.
