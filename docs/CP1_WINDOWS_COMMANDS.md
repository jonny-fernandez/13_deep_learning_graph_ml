# CP1 — Windows PowerShell commands

**Prerequisites already confirmed in user screenshots:** Python 3.12 `.venv`, basic CP1 packages, CPU-only torch and torch-geometric installed, `pip check` clean, initial starter tests passing.

Back up your existing `13_deep_learning_graph_ml` folder first. Extract the ZIP into its **parent folder** (`github files`), choosing to replace matching files; do not accidentally make a nested `13_deep_learning_graph_ml/13_deep_learning_graph_ml/`. The archive leaves your local `.env` and `.venv` alone but does not include them.

Run **one consolidated command** from the project directory:

```powershell
& .\.venv\Scripts\python.exe scripts\cp1_run.py
```

What it does:

1. Executes **all** pytest tests without network calls. Stops if a test fails.
2. Downloads Bank Marketing (~1 MB) and UCI HAR (~58 MB) from the official UCI URLs if missing. HAR may take time. PyG downloads Cora (~several MB) if missing; Internet required only for these public downloads.
3. Fits baseline logistic regression on *training only* for all three approved tasks, evaluates *validation only* (not test).
4. Writes local machine-readable evidence to `outputs/cp1/cp1_validation_report.json`. Training/evaluation times reflect your PC and are not assumed in advance.

The final test sets are sealed. CP1 does not train MLP, LSTM, Transformer, GCN or GraphSAGE — that is CP2/CP3.

**If one task fails:** do not reinstall everything. Provide the terminal error output. The script saves prior tasks' successful validation reports. After a fix, rerun the command. For individual rerun, `& .\.venv\Scripts\python.exe scripts\cp1_run.py --tasks har --skip-tests` (substitute task) is supported.

Please share the final console output **and** validation report if possible. No checkpoint is completed just by extracting the ZIP.

**Security:** do not upload `.env`; rotate any key from the earlier starter ZIP if not already revoked.

## Bank Marketing archive repair (October 16, 2026)

The original CP1 loader expected `bank-additional-full.csv` in the outer UCI ZIP.
The official UCI archive nests this CSV in `bank-additional.zip`. The
replacement loader detects both distributions without extracting arbitrary paths.
Offline regression tests cover the nested archive, malformed/duplicate members,
and safe path handling. No packages, datasets, or Python versions need reinstalling.

After replacing the project files, rerun the **same single command**:

```powershell
& .\.venv\Scripts\python.exe scripts\cp1_run.py
```

The previously downloaded `data/raw/bank_marketing/bank_marketing.zip` is cached
and will be reused. Do not delete `.venv`, `.env`, the cached ZIP, or other data
folders. Test metrics will be written only after each baseline actually finishes.

## Bank Marketing quoted-header compatibility repair (October 16, 2026)

The official nested `bank-additional-full.csv` uses a semicolon delimiter with
**quoted header names**. The earlier archive fix correctly found the CSV but
checked raw header bytes without removing CSV quoting; it rejected valid data.
This package validates headers via Python's `csv.reader`, supports UTF-8 BOM and
quoted/unquoted headers, rejects duplicate/missing required columns, and
validates cached headers. Offline regression tests cover each case.

Run the same command; the archived data and existing venv are preserved:

```powershell
& .\.venv\Scripts\python.exe scripts\cp1_run.py
```

The dataset downloads, model training, and metrics have NOT been independently
executed on your PC by packaging this fix. Provide full output and report.
