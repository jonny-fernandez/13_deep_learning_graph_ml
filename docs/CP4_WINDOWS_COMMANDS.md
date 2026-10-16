# CP4 — Two deliberate commands, not repeated replacement ZIPs

Run in PowerShell from `C:\Users\Jonny Fernandez\Documents\github files\13_deep_learning_graph_ml` using the existing Python 3.12 `.venv`. Back up the folder first, and **preserve** `.venv`, `.env`, `data/raw`, `outputs/cp1`, `outputs/cp2`, `outputs/cp3`, and all saved model checkpoints. Do not start a new repository.

**1. Read-only final-test preflight + tests (test labels remain sealed):**

```powershell
& .\.venv\Scripts\python.exe scripts\cp4_run.py
```

It runs full offline tests and `pip check`, checks CP1→CP2→CP3 report hashes, downloaded data archive hashes, all eight Cora raw-file hashes, and selected model checkpoint hashes/metadata. It writes `outputs\cp4\cp4_selection_lock.json`, without reading final-test features or labels.

**2. ONE-TIME FINAL-TEST UNLOCK, only after preflight says PASSED and frozen selection is accepted:**

```powershell
& .\.venv\Scripts\python.exe scripts\cp4_run.py --final-test
```

The second command repeats the gates, opens the sealed test labels once, compares all eight frozen-predeclared models, records per-class metrics, actual test prediction seconds, observed training duration and parameter counts, and writes:

- `outputs\cp4\cp4_final_test_report.json` (authoritative values)
- `outputs\cp4\cp4_evidence_summary.md` (copyable summary)
- `outputs\cp4\cp4_evidence_dashboard.html` (local screenshot-ready summary)

**Do not rerun the final-test command to improve performance.** It refuses to overwrite an existing final report. To regenerate presentation without opening test labels, run:

```powershell
& .\.venv\Scripts\python.exe scripts\cp4_run.py --render
```

**3. Security/commit hygiene check (no changes to source):**

```powershell
& .\.venv\Scripts\python.exe scripts\cp4_security_audit.py
```

This is a filename/.gitignore/optional-Git audit, **not** proof that source contents are free of secrets, vulnerabilities, or license concerns. Manually review staged files and dependencies before GitHub.

**4. Share with ChatGPT:** screenshot of the last console output, `cp4_final_test_report.json`, and security audit output. We will review them and establish a real screenshot list. Capture the screenshots, upload them for the **Word screenshot handoff before GitHub**, and confirm each numbered filename. Only after that can README/GitHub publication occur. Do not publish until explicitly approved.

**Limitations:** all comparisons represent a single CPU machine and run/seed; confidence intervals and inferential significance are not claimed. Test sets are used for the final report, not for redesigning a model. HAR uses different engineered/raw input representations on the same windows. Cora remains transductive. A real pipeline bug found during final evaluation is documented; any rerun must be declared, never hidden.
