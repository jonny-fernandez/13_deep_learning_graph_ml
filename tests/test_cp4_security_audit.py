"""Offline integration tests for the CP4 file-name security gate."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / 'scripts' / 'cp4_security_audit.py'
RULES = [
    '.venv/', 'venv/', '.env', 'outputs/', 'data/raw/', 'data/processed/',
    'mlruns/', 'artifacts/',
]


def project(tmp_path: Path) -> Path:
    root = tmp_path / 'p13'
    (root / 'scripts').mkdir(parents=True)
    shutil.copyfile(SRC, root / 'scripts' / 'cp4_security_audit.py')
    (root / '.gitignore').write_text('\n'.join(RULES) + '\n', encoding='utf-8')
    (root / 'README.md').write_text('Safe stub\n', encoding='utf-8')
    return root


def audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / 'scripts' / 'cp4_security_audit.py')],
        cwd=root, capture_output=True, text=True, check=False,
    )


def test_untracked_local_env_does_not_fail_before_git_init(tmp_path):
    root = project(tmp_path)
    (root / '.env').write_text('placeholder_not_a_real_secret', encoding='utf-8')
    result = audit(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'local .env file(s) excluded' in result.stdout
    assert 'placeholder_not_a_real_secret' not in result.stdout


def test_other_credential_filename_still_fails_before_git_init(tmp_path):
    root = project(tmp_path)
    (root / 'service-account.json').write_text('{}', encoding='utf-8')
    result = audit(root)
    assert result.returncode != 0
    assert 'service-account.json' in result.stderr + result.stdout


def test_missing_gitignore_env_exclusion_fails(tmp_path):
    root = project(tmp_path)
    (root / '.env').write_text('placeholder', encoding='utf-8')
    (root / '.gitignore').write_text('\n'.join(x for x in RULES if x != '.env'), encoding='utf-8')
    result = audit(root)
    assert result.returncode != 0
    assert '.gitignore missing .env exclusion' in result.stderr + result.stdout


@pytest.mark.skipif(shutil.which('git') is None, reason='Git executable unavailable')
def test_git_tracked_env_must_fail_even_when_ignored(tmp_path):
    root = project(tmp_path)
    (root / '.env').write_text('placeholder', encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', '-f', '.env'], cwd=root, check=True)
    result = audit(root)
    assert result.returncode != 0
    assert 'prohibited tracked/source paths: .env' in result.stdout + result.stderr


@pytest.mark.skipif(shutil.which('git') is None, reason='Git executable unavailable')
def test_git_untracked_env_stays_safe(tmp_path):
    root = project(tmp_path)
    (root / '.env').write_text('placeholder', encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', '.gitignore', 'README.md'], cwd=root, check=True)
    result = audit(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Git tracked-file inspection' in result.stdout


def test_env_local_remains_blocked_without_matching_ignore_rule(tmp_path):
    root = project(tmp_path)
    (root / '.env.local').write_text('placeholder', encoding='utf-8')
    result = audit(root)
    assert result.returncode != 0
    assert '.env.local' in result.stdout + result.stderr
