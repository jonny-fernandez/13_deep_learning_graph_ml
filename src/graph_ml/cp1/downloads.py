"""Small HTTPS-only dataset cache. No shell calls or arbitrary ZIP extraction."""
from __future__ import annotations
import os
import csv
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO, StringIO

BANK_URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
HAR_URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024
MAX_BANK_INNER_ZIP_BYTES = 30 * 1024 * 1024
MAX_BANK_CSV_BYTES = 30 * 1024 * 1024


def ensure_download(url: str, dest: Path) -> Path:
    if not url.startswith('https://archive.ics.uci.edu/static/public/'):
        raise ValueError('Unapproved dataset URL')
    if dest.exists():
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ValueError(f'Invalid cached archive: {dest}')
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.partial')
    try:
        with urlopen(url, timeout=90) as response, tmp.open('wb') as out:
            if not response.geturl().startswith('https://'):
                raise ValueError('Dataset download redirected to non-HTTPS URL')
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError('Download exceeds configured size limit')
                out.write(chunk)
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise ValueError('Dataset archive checksum validation failed')
        os.replace(tmp, dest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)


def _unique_member(z: ZipFile, basename: str):
    """Find exactly one named ZIP member without extracting untrusted paths."""
    matches = [item for item in z.infolist()
               if not item.is_dir() and PurePosixPath(item.filename).name == basename]
    if len(matches) != 1:
        raise ValueError(f'Expected unique {basename} in UCI archive (found {len(matches)})')
    return matches[0]


def _validate_bank_csv_header(raw: bytes) -> None:
    """Validate UCI's semicolon CSV header using CSV quoting rules.

    The official bank-additional-full.csv encodes column names as quoted
    values (e.g., '"age";"job";...'). Splitting raw bytes at semicolons
    incorrectly preserves quotes and rejects this legitimate file.
    """
    try:
        first_line = raw.split(b'\n', 1)[0].decode('utf-8-sig').rstrip('\r')
        columns = next(csv.reader(StringIO(first_line), delimiter=';', quotechar='"', strict=True))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError('Unexpected Bank Marketing CSV header') from exc
    required = {'age', 'job', 'duration', 'y'}
    if (not required.issubset(columns) or
            len(columns) != len(set(columns)) or
            any(not col or col != col.strip() for col in columns)):
        raise ValueError('Unexpected Bank Marketing CSV header')


def bank_csv(archive: Path, output: Path) -> Path:
    """Read bank-additional-full.csv from direct or nested UCI ZIP structure.

    The official static/public/222/bank+marketing.zip contains a nested
    bank-additional.zip, which in turn contains bank-additional-full.csv.
    Other official distribution variants can hold the CSV directly.
    Only the expected member's bytes are read; nothing is extracted by path.
    """
    if output.exists():
        if not output.is_file() or not 0 < output.stat().st_size <= MAX_BANK_CSV_BYTES:
            raise ValueError(f'Invalid cached Bank Marketing CSV: {output}')
        with output.open('rb') as cache:
            _validate_bank_csv_header(cache.readline())
        return output

    with ZipFile(archive) as outer:
        direct = [item for item in outer.infolist() if not item.is_dir()
                  and PurePosixPath(item.filename).name == 'bank-additional-full.csv']
        if len(direct) > 1:
            raise ValueError('Expected unique bank-additional-full.csv in UCI archive')
        if direct:
            member = direct[0]
            if member.file_size > MAX_BANK_CSV_BYTES:
                raise ValueError('Bank Marketing CSV exceeds expected size limit')
            raw = outer.read(member)
        else:
            nested = _unique_member(outer, 'bank-additional.zip')
            if nested.file_size > MAX_BANK_INNER_ZIP_BYTES:
                raise ValueError('Nested Bank Marketing ZIP exceeds expected size limit')
            with ZipFile(BytesIO(outer.read(nested))) as inner:
                member = _unique_member(inner, 'bank-additional-full.csv')
                if member.file_size > MAX_BANK_CSV_BYTES:
                    raise ValueError('Bank Marketing CSV exceeds expected size limit')
                raw = inner.read(member)

    # Fail before caching malformed content. Accept UCI's quoted headers.
    _validate_bank_csv_header(raw)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.partial')
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def prepare_har(archive: Path, output: Path) -> Path:
    """Extract ONLY train data and test subject identifiers, not test labels/features."""
    required = [f'train/{n}' for n in ('X_train.txt', 'y_train.txt', 'subject_train.txt')]
    required += ['test/subject_test.txt']
    required += [f'train/Inertial Signals/{signal}_{axis}_train.txt'
                 for signal in ('total_acc', 'body_acc', 'body_gyro') for axis in ('x', 'y', 'z')]
    if all((output / item).exists() for item in required):
        return output
    with ZipFile(archive) as outer:
        nested = [n for n in outer.namelist() if n.endswith('UCI HAR Dataset.zip')]
        if nested:
            with ZipFile(BytesIO(outer.read(nested[0]))) as inner:
                _extract_allowlist(inner, required, output)
        else:
            _extract_allowlist(outer, required, output)
    return output


def _extract_allowlist(z: ZipFile, items: list[str], output: Path) -> None:
    for member in items:
        matches = [n for n in z.namelist() if n.endswith('UCI HAR Dataset/' + member)]
        if len(matches) != 1:
            raise ValueError(f'HAR archive missing or duplicated required file: {member}')
        dest = output / member
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(matches[0]) as source, dest.open('wb') as target:
            # All requested files are known UCI data with bounded sizes.
            from shutil import copyfileobj
            if z.getinfo(matches[0]).file_size > 80 * 1024 * 1024:
                raise ValueError('Unexpectedly large HAR member')
            copyfileobj(source, target)
