from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tools.version import (
    archive_changelog,
    check,
    core_pin_version,
    project_version,
    release,
    set_version,
    validate_version,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_versions_are_synchronized() -> None:
    version = project_version(ROOT)
    # Manifest + circuitopt-core pin synchronization is a repo invariant.
    assert check(ROOT) == []
    # A tag matching the project version must not raise a tag-mismatch. The
    # CHANGELOG-archived part of the --tag check is a release-time gate (see
    # test_release_updates_every_manifest); a `set`-but-not-`release`d version
    # (e.g. a pre-release rc) intentionally keeps its entry under [Unreleased],
    # so we do not require the version to be archived here.
    tag_errors = check(ROOT, tag=f"v{version}")
    assert not any(e.startswith("release tag") for e in tag_errors)


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


def _write_fake_repo(tmp_path: Path, version: str) -> None:
    """Create the minimal manifest tree that ``synchronized_content`` reads."""
    frontend = tmp_path / "frontend"
    tauri = frontend / "src-tauri"
    rust = tmp_path / "rust"
    tauri.mkdir(parents=True)
    rust.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{version}"\n'
        f'dependencies = ["numpy>=2.0", "circuitopt-core=={version}"]\n',
        encoding="utf-8",
    )
    (frontend / "package.json").write_text(
        f'{{"name": "example", "version": "{version}"}}\n',
        encoding="utf-8",
    )
    (frontend / "package-lock.json").write_text(
        f'{{"name": "example", "version": "{version}", '
        f'"packages": {{"": {{"version": "{version}"}}}}}}\n',
        encoding="utf-8",
    )
    (tauri / "Cargo.toml").write_text(
        f'[package]\nname = "example"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (tauri / "tauri.conf.json").write_text(
        f'{{"productName": "Example", "version": "{version}"}}\n',
        encoding="utf-8",
    )
    # Mirrors rust/Cargo.toml: the version lives in [workspace.package] only
    # (member crates use `version.workspace = true`, which must NOT be matched).
    (rust / "Cargo.toml").write_text(
        "[workspace]\n"
        'resolver = "3"\n'
        'members = ["crates/example"]\n'
        "\n"
        "[workspace.package]\n"
        f'version = "{version}"\n'
        'edition = "2024"\n'
        'license = "MIT"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"""# Changelog

## [Unreleased] / 未发布

- A new feature.

## [{version}] - 2026-07-17

[Unreleased]: https://example.test/compare/v{version}...HEAD
[{version}]: https://example.test/releases/v{version}
""",
        encoding="utf-8",
    )


def test_release_updates_every_manifest(tmp_path: Path) -> None:
    _write_fake_repo(tmp_path, "1.3.0")
    frontend = tmp_path / "frontend"
    tauri = frontend / "src-tauri"

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
    rust_cargo = (tmp_path / "rust/Cargo.toml").read_text(encoding="utf-8")
    assert 'version = "1.4.0"' in rust_cargo
    # Only the [workspace.package] assignment may be rewritten; the rest of the
    # manifest (members list, edition, license) must round-trip untouched.
    assert 'members = ["crates/example"]' in rust_cargo
    assert 'edition = "2024"' in rust_cargo


def test_check_catches_rust_workspace_drift(tmp_path: Path) -> None:
    _write_fake_repo(tmp_path, "1.4.0")
    rust_cargo = tmp_path / "rust/Cargo.toml"
    rust_cargo.write_text(
        rust_cargo.read_text(encoding="utf-8").replace(
            'version = "1.4.0"', 'version = "1.3.0"'
        ),
        encoding="utf-8",
    )

    errors = check(tmp_path)

    assert any("rust/Cargo.toml" in error for error in errors)


def test_repository_pins_circuitopt_core_to_project_version() -> None:
    # D9: the real pyproject pins the compiled Rust core exactly to the project.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert core_pin_version(text) == project_version(ROOT)


def test_check_catches_core_pin_drift(tmp_path: Path) -> None:
    _write_fake_repo(tmp_path, "1.4.0")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "circuitopt-core==1.4.0", "circuitopt-core==1.3.0"
        ),
        encoding="utf-8",
    )

    errors = check(tmp_path)

    assert any("circuitopt-core pin" in error for error in errors)


def test_set_version_bumps_core_pin(tmp_path: Path) -> None:
    _write_fake_repo(tmp_path, "1.4.0")
    set_version("2.0.0-rc1", tmp_path)

    text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert core_pin_version(text) == "2.0.0-rc1"
    assert check(tmp_path) == []
