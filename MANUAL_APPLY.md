# Manual application

The guarded installer is recommended:

```bash
python apply_comnet_update.py --repo /path/to/iot-trust-compression
python apply_comnet_update.py --repo /path/to/iot-trust-compression --apply --run-tests
```

The first command is a dry run.

To apply manually, copy the package files to matching repository paths, then make the four guarded text changes described in `PATCH_NOTES.md`. Do not copy, open, or commit a credential file.

Afterwards:

```bash
python -m py_compile src/crux.py src/diagnostic.py src/comnet_audit.py src/mitigate.py
git diff --check
python -m pytest -q tests/test_compat_modules.py tests/test_comnet_audit.py
```

Review with:

```bash
git status --short
git diff -- README.md .gitignore SECURITY.md REPRODUCIBILITY.md \
  COMPUTER_NETWORKS_RUN_ORDER.md src tests notebooks config/config.yaml
```
