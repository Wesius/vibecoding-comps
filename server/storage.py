from __future__ import annotations

import ast
import io
import shutil
import zipfile
from pathlib import Path


def validate_and_store_agent(
    name: str,
    zip_bytes: bytes,
    agents_dir: Path,
    max_bytes: int,
) -> Path:
    """Validate an agent zip and extract to disk.

    Returns the path to the extracted agent directory.
    Raises ValueError on validation failure.
    """
    if len(zip_bytes) > max_bytes:
        raise ValueError(
            f"Zip file too large ({len(zip_bytes)} bytes, max {max_bytes})."
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise ValueError("Invalid zip file.")

    # Find agent.py in the zip
    names = zf.namelist()
    agent_py = None

    # Check root level
    if "agent.py" in names:
        agent_py = "agent.py"
    else:
        # Check one level deep (in case zip has a top-level directory)
        for n in names:
            parts = Path(n).parts
            if len(parts) == 2 and parts[1] == "agent.py":
                agent_py = n
                break

    if agent_py is None:
        raise ValueError(
            "Zip must contain agent.py at the root or in a single top-level directory."
        )

    # AST-validate agent.py (no execution)
    source = zf.read(agent_py).decode("utf-8")
    _validate_agent_source(source)

    # Extract to agents_dir/name/
    dest = agents_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Extract all .py files, flattening if needed
    prefix = str(Path(agent_py).parent)
    for info in zf.infolist():
        if info.is_dir():
            continue
        if not info.filename.endswith(".py"):
            continue

        # Compute relative path
        rel = info.filename
        if prefix and prefix != ".":
            if rel.startswith(prefix + "/"):
                rel = rel[len(prefix) + 1:]
            else:
                continue

        out_path = dest / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(info.filename))

    return dest


def _validate_agent_source(source: str) -> None:
    """AST-parse agent.py to check for required structure."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in agent.py: {e}")

    # Look for class Agent with on_tick method
    found_class = False
    found_method = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Agent":
            found_class = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "on_tick":
                        found_method = True
                        break

    if not found_class:
        raise ValueError("agent.py must contain a class named 'Agent'.")
    if not found_method:
        raise ValueError("Agent class must have an 'on_tick' method.")
