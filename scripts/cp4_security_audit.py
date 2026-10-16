"""Read-only Project 13 credential-filename and Git tracking safety gate.

Never reads or prints the contents of local secrets. A local .env excluded by
.gitignore is acceptable before git init; an *already tracked* .env is not.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    '.env', '.env.local', '.env.production', 'credentials.json',
    'service-account.json', 'id_rsa', 'id_ed25519',
}
IGNORED = [
    '.venv/', 'venv/', '.env', 'outputs/', 'data/raw/', 'data/processed/',
    'mlruns/', 'artifacts/',
]
EXCLUDED_DIRS = {
    '.git', '.venv', 'venv', 'outputs', 'data', 'artifacts',
    'logs', '__pycache__', '.pytest_cache',
}
FORBIDDEN_PREFIXES = (
    '.venv/', 'venv/', 'outputs/', 'data/raw/', 'data/processed/',
)


def _has_valid_gitignore() -> None:
    ignore_path = ROOT / '.gitignore'
    if not ignore_path.is_file():
        raise SystemExit('FAIL: project .gitignore missing')
    rules = set(ignore_path.read_text(encoding='utf-8').splitlines())
    for rule in IGNORED:
        if rule not in rules and rule.rstrip('/') + '/*' not in rules:
            raise SystemExit(f'FAIL: .gitignore missing {rule} exclusion')


def _tracked_or_local_source_files() -> tuple[list[str], str, list[str]]:
    """Use tracked files when Git exists; otherwise emulate source selection.

    The only special local-filename exemption is exact `.env` because the
    preceding gate ensures it is excluded by .gitignore. With a Git repo,
    Git's *actual index* is authoritative, so tracked .env still fails.
    """
    try:
        detected = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        detected = None
    if detected is not None and detected.returncode == 0 and detected.stdout.strip() == 'true':
        result = subprocess.run(
            ['git', 'ls-files', '-z'], cwd=ROOT,
            capture_output=True, check=True,
        )
        files = [s for s in result.stdout.decode('utf-8', errors='replace').split('\0') if s]
        return files, 'Git tracked-file inspection', []

    files: list[str] = []
    local_ignored_env: list[str] = []
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        name = relative.as_posix()
        if path.name == '.env':
            # It is explicitly excluded by the mandatory .gitignore rule.
            # This is not evidence that any historical copy is safe.
            local_ignored_env.append(name)
            continue
        files.append(name)
    return files, 'Working source-tree inspection (Git not initialized)', local_ignored_env


def main() -> None:
    _has_valid_gitignore()
    files, mode, ignored_env = _tracked_or_local_source_files()
    bad = [
        name for name in files
        if Path(name).name.lower() in FORBIDDEN or name.startswith(FORBIDDEN_PREFIXES)
    ]
    if bad:
        raise SystemExit('FAIL prohibited tracked/source paths: ' + ', '.join(bad[:10]))
    print(f'PASS: {mode}; {len(files)} candidate files; no prohibited tracked/source filenames.')
    if ignored_env:
        print(f'NOTE: {len(ignored_env)} local .env file(s) excluded under .gitignore; never commit or distribute.')
    print('Manual secret content review and data/license audit are STILL REQUIRED before commit.')
    print('Audit did not open or print any .env contents, credentials, or test labels.')


if __name__ == '__main__':
    main()
