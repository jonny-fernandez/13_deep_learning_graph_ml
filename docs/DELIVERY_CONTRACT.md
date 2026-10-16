# Delivery Contract — Minimal Replacement Cycles

Purpose: avoid the Project 12 pattern of many sequential source-file replacements while retaining real validation.

## For each of the 4 checkpoint groups

1. Review the current starter and all user-supplied command outputs **before** modifying code.
2. Implement all related source/config/doc/test changes locally as a group.
3. Deliver **one self-contained ZIP** with the full updated project tree, an exact change manifest and a single first validation command, plus any necessary follow-up diagnostics.
4. User runs commands and shares unedited output. A checkpoint passes only if its exit gates are evidenced.
5. If an issue appears, diagnose the evidence first; resolve the batch of related defects and issue one corrected ZIP as needed. Do not stage minor changes over 12–25 individual file updates.
6. Don't bundle unfinished code under a completed checkpoint label. Explicitly distinguish *planned*, *implemented awaiting user validation*, and *verified*.

## Initial state

This refreshed ZIP **only updates documentation and supplies a read-only CP1 inspection utility**. Existing ML implementation stubs and the initial test files are preserved intact. CP1–CP4 remain pending.

## Safety before replacing local folders

- Back up the existing project or use Git before replacing it with the package.
- The archive deliberately omits `.env`, caches and compiled Python files. Preserve local env/settings securely in your own environment; do not copy an exposed key to a new archive.
- Unzip over the matching project folder, not into a second nested project folder; review any local changes before overwriting.
- The first CP1 inspection command is `python scripts/cp1_readonly_audit.py`.

## No silent scope reduction

Original 16 original tasks are all mapped in `CHECKPOINTS.md`; condensing checkpoints changes **packaging and validation gates**, not the scientific questions or requested model families.
