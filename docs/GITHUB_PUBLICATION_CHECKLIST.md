# GitHub Publication — Final Local Review (PowerShell)

**Current verified stage:** Eight PNG screenshots saved in `screenshots/` and a Word evidence handoff supplied. CP4 final evaluation has completed, but no public Project 13 repository or successful GitHub Actions has been verified. Follow these steps; **stop before push** and share staged-file review output for approval.

## 1. Before Git

From PowerShell in `C:\Users\Jonny Fernandez\Documents\github files\13_deep_learning_graph_ml`:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe scripts\cp4_security_audit.py
```

Do NOT run the `--final-test` command again; its one-time report is already complete. Ensure the API key exposed in a previously shared starter ZIP has been revoked. Check README accuracy, source files for secrets and third-party licensing. Automated path inspection alone does not detect all secrets or vulnerabilities.

## 2. Initialize Git, without pushing

Run this exactly from the Project 13 folder **only if it has no existing `.git` directory**:

```powershell
git init -b main
git add .github .gitignore README.md CHECKPOINTS.md PROJECT_PLAN.md PROJECT_RULES.md PACKAGE_MANIFEST.md UPDATES_FROM_JOB_REVIEW.md configs docs scripts src tests screenshots pyproject.toml requirements.txt requirements-cp1.txt
git status --short
git diff --cached --check
git ls-files -- .env
```

`git ls-files -- .env` should print **nothing**. `.env` should remain ignored locally; the eight PNG screenshots and `screenshots/SCREENSHOT_MANIFEST.txt` should appear in `git status --short`. All `data/raw`, `.venv`, `outputs/cp*`, and model checkpoints should remain outside Git. The above `git add` uses an explicit source/documentation allowlist rather than staging the entire directory; do not add unreviewed files.

If Git was **already** initialized, do not reinitialize; run `git status --short` to inspect it first. Do not paste secret values into chat; only send non-sensitive filenames/status and the final PASS/FAIL lines.

## 3. Manually inspect what will be committed

```powershell
git diff --cached --stat
git ls-files
git diff --cached --check
```

Check that no `.env`, keys, private datasets, model binaries, local experiment reports or other sensitive material are staged. If an unwanted file is staged, use `git restore --staged -- "path/to/file"` rather than deleting your local data.

**Stop here and share the output for review.** No push, successful CI, or remote repository has yet been established by these commands.

## 4. After approval, publish and verify (not yet performed)

1. Create an **empty** public repository named `13_deep_learning_graph_ml` in GitHub, without initializing a separate README or Git history.
2. Ensure your Git/GitHub credentials are configured, then run `git commit -m "Complete Project 13 evaluation platform"`.
3. Add the real repository's HTTPS URL as `origin`, e.g. `git remote add origin https://github.com/<your-username>/13_deep_learning_graph_ml.git`; substitute your own confirmed account/repository URL.
4. Run `git push -u origin main` and check the repository tree on GitHub.
5. Open **Actions** to confirm the *Python Regression & Repository Hygiene* job actually completed successfully. If it fails, capture actual logs, fix the failure, retest locally, and push a corrective commit.
6. Only then capture a **real** `09_github_actions_verified.png` supplemental evidence screenshot and mark CP4/Project 13 complete; resume claims must use only verified results.

If you want an explicit software reuse license, choose it deliberately before publication; the original source ZIP did not include a standalone Project 13 license. Upstream dataset code licenses are not a substitute for that decision.
