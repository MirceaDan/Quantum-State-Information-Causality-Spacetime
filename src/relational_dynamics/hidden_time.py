"""Static audit separating computational controls from physical time variables."""

from __future__ import annotations

import ast
from pathlib import Path


_RELEVANT = {"t", "time", "physical_time", "external_time", "timestep", "step", "iteration", "computational_index", "max_iterations"}


def _classification(name: str) -> tuple[str, bool]:
    if name in {"t", "time", "physical_time"}:
        return "physical_time", False
    if name in {"external_time", "computational_index"}:
        return "metadata_label", True
    return "numerical_control", True


def audit_source_tree(source_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            for name in names:
                if name in _RELEVANT:
                    role, allowed = _classification(name)
                    records.append({
                        "file": str(path),
                        "line": node.lineno,
                        "variable": name,
                        "role": role,
                        "allowed": allowed,
                    })
    return records
