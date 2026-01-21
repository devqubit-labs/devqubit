# Configuration file for Sphinx.
# See Sphinx docs for details:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path


project = "devqubit"
author = "devqubit"
copyright = f"{date.today().year}, {author}"


def _version_from_pyproject() -> str | None:
    """Reads version from repository's pyproject.toml."""

    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    # PEP 621
    proj = data.get("project", {})
    v = proj.get("version")
    if isinstance(v, str) and v.strip():
        return v.strip()

    # Poetry fallback (optional)
    poetry = data.get("tool", {}).get("poetry", {})
    v = poetry.get("version")
    if isinstance(v, str) and v.strip():
        return v.strip()

    return None


# Read package version (robust fallback)
release = _version_from_pyproject() or "0.0.0"
version = release.split("+")[0]

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# Make section labels unique across pages
autosectionlabel_prefix_document = True

# MyST (Markdown) configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3  # auto-generate anchors for h1-h3

# Treat ```mermaid fences as Sphinx directives
myst_fence_as_directive = ["mermaid"]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# Show "Edit on GitHub"
html_context = {
    "display_github": True,
    "github_user": "devqubit-labs",
    "github_repo": "devqubit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
