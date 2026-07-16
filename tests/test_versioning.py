from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tools.version import (
    archive_changelog,
    check,
    project_version,
    release,
    validate_version,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_versions_are_synchronized() -> None:
    version = project_version(ROOT)
    assert check(ROOT) == []
    assert check(ROOT, tag=f"v{version}") == []


@pytest.mark.parametrize(
    "version",
    ["1", "1.2", "v1.2.3", "01.2.3", "1.2.3.4", "1.2 latest"],
)
def test_validate_version_rejects_non_semver(version: str) -> None:
    with pytest.raises(ValueError, match="invalid semantic version"):
        validate_version(version)


def test_archive_changelog_moves_unreleased_content_to_new_release() -> None:
    changelog = """# Changelog

## [Unreleased] / 未发布

### Added / 新增

- A new feature.

## [1.3.0] - 2026-07-17

- Previous release.

[Unreleased]: https://example.test/compare/v1.3.0...HEAD
[1.3.0]: https://example.test/releases/v1.3.0
"""

    updated = archive_changelog(changelog, "1.4.0", dt.date(2026, 7, 18).isoformat())

    assert "## [Unreleased] / 未发布\n\n## [1.4.0] - 2026-07-18" in updated
    assert updated.index("## [1.4.0]") < updated.index("### Added / 新增")
    assert (
        "[Unreleased]: https://github.com/751K/circuit-optimization-lab/"
        "compare/v1.4.0...HEAD"
    ) in updated
    assert (
        "[1.4.0]: https://github.com/751K/circuit-optimization-lab/"
        "compare/v1.3.0...v1.4.0"
    ) in updated


def test_release_updates_every_manifest(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    tauri = frontend / "src-tauri"
    tauri.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "1.3.0"\n',
        encoding="utf-8",
    )
    (frontend / "package.json").write_text(
        '{"name": "example", "version": "1.3.0"}\n',
        encoding="utf-8",
    )
    (frontend / "package-lock.json").write_text(
        '{"name": "example", "version": "1.3.0", '
        '"packages": {"": {"version": "1.3.0"}}}\n',
        encoding="utf-8",
    )
    (tauri / "Cargo.toml").write_text(
        '[package]\nname = "example"\nversion = "1.3.0"\n',
        encoding="utf-8",
    )
    (tauri / "tauri.conf.json").write_text(
        '{"productName": "Example", "version": "1.3.0"}\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        """# Changelog

## [Unreleased] / 未发布

- A new feature.

## [1.3.0] - 2026-07-17

[Unreleased]: https://example.test/compare/v1.3.0...HEAD
[1.3.0]: https://example.test/releases/v1.3.0
""",
        encoding="utf-8",
    )

    release("1.4.0", "2026-07-18", tmp_path)

    assert project_version(tmp_path) == "1.4.0"
    assert check(tmp_path, tag="v1.4.0") == []
    assert (
        '"version": "1.4.0"'
        in (frontend / "package.json").read_text(encoding="utf-8")
    )
    assert (
        (frontend / "package-lock.json")
        .read_text(encoding="utf-8")
        .count('"version": "1.4.0"')
        == 2
    )
    assert (
        'version = "1.4.0"'
        in (tauri / "Cargo.toml").read_text(encoding="utf-8")
    )
    assert (
        '"version": "1.4.0"'
        in (tauri / "tauri.conf.json").read_text(encoding="utf-8")
    )
