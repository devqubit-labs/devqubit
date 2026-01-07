# Configuration file for Sphinx.
# See Sphinx docs for details:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version


project = "devqubit"
author = "devqubit"
copyright = f"{date.today().year}, {author}"

# Read package version
release = "0.0.0"
try:
    release = pkg_version("devqubit")
except PackageNotFoundError:
    pass

version = release.split("+")[0]

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# Make section labels unique across pages
autosectionlabel_prefix_document = True

# MyST (Markdown) configuration
myst_enable_extensions = [
    "colon_fence",  # ::: fences for admonitions
    "deflist",  # definition lists
    "tasklist",
]
myst_heading_anchors = 3  # auto-generate anchors for h1-h3

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = []

# Show "Edit on GitHub"
html_context = {
    "display_github": True,
    "github_user": "devqubit-labs",
    "github_repo": "devqubit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}
