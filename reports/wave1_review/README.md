# Wave-one review artifacts

The maintained interpretation and experiment queue live in `docs/EXPERIMENT_JOURNAL.md` at the Clinical project root. This directory contains reproducible evidence for that journal.

- `summary.json`: nine verified evaluations, training registry, fixed subgroups, token/runtime information, local adapter checks, and paired bootstrap comparisons.
- `case_evidence.json`: source records, original gold targets, and three models' outputs for 29 reviewed validation cases. Includes all 20 numeric answers marked incorrect for the selected checkpoint. This is an evidence packet, not adjudicated labels.
- `input_manifest.json`: hashes for analysis inputs and local adapter weights. The script hash identifies the analysis implementation.
- `huggingface_verification.json`: read-only verification of nine uploaded adapters and adapter configurations at pinned repository revisions. The original verification snapshot remains in place when the audit is rerun without `--verify-hf`.

From the Clinical project root:

```bash
.venv/bin/python scripts/review_wave1.py
.venv/bin/python scripts/review_wave1.py --verify-hf
```

The second command requires access to the existing private Hugging Face repositories. It uses the configured login, never prints credentials, downloads only adapter configuration files into temporary storage, and compares remote LFS weight hashes with local bytes. It does not upload or modify models.

No new generations, corrected headline scores, new training runs, or clinical adjudications are claimed by these artifacts. Bootstrap intervals concern the original scorer's outputs and do not correct its documented limitations. The test outputs have been reviewed and cannot serve as an untouched holdout for wave-two development.
