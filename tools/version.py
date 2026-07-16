#!/usr/bin/env python3
"""Synchronize project versions from the canonical ``pyproject.toml`` value."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def validate_version(version: str) -> str:
    version = str(version).strip()
    if not SEMVER.fullmatch(version):
        raise ValueError(f"invalid semantic version: {version!r}")
    return version


def project_version(root: Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'(?ms)^\[project\]\s*.*?^version\s*=\s*"([^"]+)"',
        text,
    )
    if not match:
        raise ValueError("could not locate [project].version in pyproject.toml")
    return validate_version(match.group(1))


def _replace_project_version(text: str, version: str) -> str:
    pattern = re.compile(
        r'(?ms)(^\[project\]\s*.*?^version\s*=\s*")[^"]+(")')
    updated, count = pattern.subn(
        lambda match: match.group(1) + version + match.group(2),
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("could not locate [project].version in pyproject.toml")
    return updated


def _json_with_version(text: str, version: str, *, lockfile: bool = False) -> str:
    data = json.loads(text)
    data["version"] = version
    if lockfile:
        package = data.get("packages", {}).get("")
        if not isinstance(package, dict):
            raise ValueError("package-lock.json has no root package entry")
        package["version"] = version
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _cargo_with_version(text: str, version: str) -> str:
    pattern = re.compile(
        r'(?ms)(^\[package\]\s*.*?^version\s*=\s*")[^"]+(")')
    updated, count = pattern.subn(
        lambda match: match.group(1) + version + match.group(2),
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("could not locate [package].version in Cargo.toml")
    return updated


def synchronized_content(root: Path, version: str) -> dict[Path, str]:
    version = validate_version(version)
    package_json = root / "frontend/package.json"
    package_lock = root / "frontend/package-lock.json"
    cargo_toml = root / "frontend/src-tauri/Cargo.toml"
    tauri_json = root / "frontend/src-tauri/tauri.conf.json"
    return {
        package_json: _json_with_version(
            package_json.read_text(encoding="utf-8"),
            version,
        ),
        package_lock: _json_with_version(
            package_lock.read_text(encoding="utf-8"),
            version,
            lockfile=True,
        ),
        cargo_toml: _cargo_with_version(
            cargo_toml.read_text(encoding="utf-8"),
            version,
        ),
        tauri_json: _json_with_version(
            tauri_json.read_text(encoding="utf-8"),
            version,
        ),
    }


def sync(root: Path = ROOT, version: str | None = None) -> list[Path]:
    version = project_version(root) if version is None else validate_version(version)
    changed = []
    for path, expected in synchronized_content(root, version).items():
        if path.read_text(encoding="utf-8") != expected:
            path.write_text(expected, encoding="utf-8")
            changed.append(path)
    return changed


def set_version(version: str, root: Path = ROOT) -> list[Path]:
    version = validate_version(version)
    pyproject = root / "pyproject.toml"
    current = pyproject.read_text(encoding="utf-8")
    updated = _replace_project_version(current, version)
    changed = []
    if current != updated:
        pyproject.write_text(updated, encoding="utf-8")
        changed.append(pyproject)
    changed.extend(sync(root, version))
    return changed


def latest_changelog_version(text: str) -> str:
    match = re.search(
        r"(?m)^## \[(?!Unreleased\])([^\]]+)\] - \d{4}-\d{2}-\d{2}$",
        text,
    )
    if not match:
        raise ValueError("CHANGELOG.md has no released version heading")
    return validate_version(match.group(1))


def archive_changelog(text: str, version: str, date: str) -> str:
    version = validate_version(version)
    try:
        dt.date.fromisoformat(date)
    except ValueError as exc:
        raise ValueError(f"invalid release date: {date!r}") from exc
    previous = latest_changelog_version(text)
    if re.search(rf"(?m)^## \[{re.escape(version)}\] ", text):
        raise ValueError(f"CHANGELOG.md already contains {version}")
    marker = "## [Unreleased] / 未发布\n"
    if marker not in text:
        raise ValueError("CHANGELOG.md has no Unreleased heading")
    text = text.replace(
        marker,
        marker + f"\n## [{version}] - {date}\n",
        1,
    )
    unreleased_link = re.compile(r"(?m)^\[Unreleased\]: .+$")
    replacement = (
        f"[Unreleased]: https://github.com/751K/circuit-optimization-lab/"
        f"compare/v{version}...HEAD\n"
        f"[{version}]: https://github.com/751K/circuit-optimization-lab/"
        f"compare/v{previous}...v{version}"
    )
    text, count = unreleased_link.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError("CHANGELOG.md has no Unreleased comparison link")
    return text


def release(version: str, date: str, root: Path = ROOT) -> list[Path]:
    version = validate_version(version)
    changelog = root / "CHANGELOG.md"
    current = changelog.read_text(encoding="utf-8")
    updated = archive_changelog(current, version, date)
    changed = set_version(version, root)
    changelog.write_text(updated, encoding="utf-8")
    return [changelog, *changed]


def check(root: Path = ROOT, tag: str | None = None) -> list[str]:
    version = project_version(root)
    errors = []
    for path, expected in synchronized_content(root, version).items():
        if path.read_text(encoding="utf-8") != expected:
            errors.append(f"{path.relative_to(root)} is not synchronized to {version}")
    if tag is not None and tag != f"v{version}":
        errors.append(f"release tag {tag!r} does not match project version v{version}")
    if tag is not None:
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        if latest_changelog_version(changelog) != version:
            errors.append(f"latest changelog release is not {version}")
    return errors


def _print_changed(paths: list[Path], root: Path = ROOT) -> None:
    if not paths:
        print("version files already synchronized")
        return
    for path in dict.fromkeys(paths):
        print(path.relative_to(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage all project versions from pyproject.toml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="print the canonical project version")
    check_parser = subparsers.add_parser(
        "check",
        help="fail when synchronized manifests disagree",
    )
    check_parser.add_argument("--tag", help="also validate a release tag such as v1.4.0")
    subparsers.add_parser(
        "sync",
        help="synchronize frontend and Tauri manifests",
    )
    set_parser = subparsers.add_parser(
        "set",
        help="set pyproject.toml and synchronize manifests",
    )
    set_parser.add_argument("version")
    release_parser = subparsers.add_parser(
        "release",
        help="set the version and archive Unreleased changelog entries",
    )
    release_parser.add_argument("version")
    release_parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="YYYY-MM-DD",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "show":
            print(project_version())
        elif args.command == "check":
            errors = check(tag=args.tag)
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print(f"version {project_version()} is synchronized")
        elif args.command == "sync":
            _print_changed(sync())
        elif args.command == "set":
            _print_changed(set_version(args.version))
        elif args.command == "release":
            _print_changed(release(args.version, args.date))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
