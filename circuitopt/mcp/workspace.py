"""Workspace-confined file access for MCP tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable


class WorkspaceError(ValueError):
    """A requested path is outside the configured MCP workspace."""


class Workspace:
    """Resolve MCP paths without permitting absolute paths or traversal."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {self.root}")

    def resolve_input(
        self,
        relative_path: str,
        *,
        suffixes: Iterable[str] = (),
    ) -> Path:
        path = self._resolve_relative(relative_path)
        if not path.is_file():
            raise WorkspaceError(f"file not found in workspace: {relative_path}")
        allowed = tuple(str(value).lower() for value in suffixes)
        if allowed and path.suffix.lower() not in allowed:
            raise WorkspaceError(
                f"file {relative_path!r} must use one of: {', '.join(allowed)}")
        return path

    def artifact_path(self, kind: str) -> tuple[Path, str]:
        safe_kind = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in kind
        ).strip("-") or "result"
        relative = Path("results") / "mcp" / (
            f"{safe_kind}-{uuid.uuid4().hex[:12]}.json"
        )
        path = self._resolve_relative(relative.as_posix())
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, relative.as_posix()

    def write_json_artifact(
        self,
        kind: str,
        payload: Any,
    ) -> str:
        path, relative = self.artifact_path(kind)
        write_json_atomic(path, payload)
        return relative

    def _resolve_relative(self, relative_path: str) -> Path:
        raw = str(relative_path).strip()
        if not raw:
            raise WorkspaceError("path must be a non-empty workspace-relative path")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise WorkspaceError("absolute paths are not allowed")
        if ".." in candidate.parts:
            raise WorkspaceError("parent path components ('..') are not allowed")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(
                f"path escapes the configured workspace: {relative_path}"
            ) from exc
        return resolved


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write strict JSON beside *path*, then atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
