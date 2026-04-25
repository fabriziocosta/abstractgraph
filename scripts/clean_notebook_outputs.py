"""Clear or check Jupyter notebook outputs.

Use this before committing notebooks so large rendered images do not end up in
Git history. With --check, the command fails if any tracked notebook contains
outputs or execution counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _notebook_has_outputs(notebook: dict) -> bool:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            return True
        if cell.get("outputs"):
            return True
    return False


def _clear_notebook(notebook: dict) -> bool:
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    metadata = notebook.get("metadata", {})
    if "widgets" in metadata:
        metadata.pop("widgets")
        changed = True
    return changed


def _iter_notebook_paths(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for path in paths:
        if path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            notebooks.append(path)
    return notebooks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("notebooks")])
    parser.add_argument("--check", action="store_true", help="fail if outputs are present")
    args = parser.parse_args()

    dirty: list[Path] = []
    for path in _iter_notebook_paths(args.paths):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if args.check:
            if _notebook_has_outputs(notebook):
                dirty.append(path)
            continue
        if _clear_notebook(notebook):
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            dirty.append(path)

    if args.check and dirty:
        print("Notebook outputs must be cleared before commit:")
        for path in dirty:
            print(f"  {path}")
        print("Run: python scripts/clean_notebook_outputs.py")
        return 1

    if dirty and not args.check:
        for path in dirty:
            print(f"cleared {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
