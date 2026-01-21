# Configuration file for Sphinx.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path


# -- Project information -----------------------------------------------------

project = "devqubit"
author = "devqubit"
copyright = f"{date.today().year}, {author}"

# Try to read version from pyproject.toml (optional)
ROOT = Path(__file__).resolve().parents[1]
_pyproject = ROOT / "pyproject.toml"
release = ""
version = ""
if _pyproject.exists():
    try:
        data = tomllib.loads(_pyproject.read_text(encoding="utf-8"))
        release = str(data.get("project", {}).get("version", "")) or ""
        version = release
    except Exception:
        release = ""
        version = ""


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
]
myst_fence_as_directive = {
    "mermaid",
}


# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_css_files = [
    "css/mermaid.css",
]

# Show "Edit on GitHub"
html_context = {
    "display_github": True,
    "github_user": "devqubit-labs",
    "github_repo": "devqubit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


# -- Mermaid (sphinxcontrib-mermaid) ----------------------------------------

mermaid_output_format = "raw"  # keep diagrams interactive in HTML

# Nice UX on ReadTheDocs
mermaid_d3_zoom = True
mermaid_fullscreen = True
mermaid_fullscreen_button_opacity = 35

# Global Mermaid "house style" (applies to all diagrams)
_mermaid_config = {
    "startOnLoad": True,
    "securityLevel": "loose",
    "theme": "base",
    "flowchart": {
        "curve": "basis",
        "nodeSpacing": 36,
        "rankSpacing": 44,
        "htmlLabels": True,
    },
    "themeVariables": {
        "fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif",
        "fontSize": "14px",
        "background": "#ffffff",
        "textColor": "#0f172a",
        "lineColor": "#94a3b8",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "edgeLabelBackground": "#ffffff",
    },
}

mermaid_init_js = f"mermaid.initialize({json.dumps(_mermaid_config)});"
