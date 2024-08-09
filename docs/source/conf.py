# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path
from typing import Any

from recommonmark.parser import CommonMarkParser

for x in os.walk("../../src"):
    sys.path.insert(0, x[0])

project = "BlueCast"
copyright = "2024, Thomas Meißner"
author = "Thomas Meißner"
release = "1.5.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

source_suffix = [".rst", ".md"]
source_parsers = {
    ".md": CommonMarkParser,
}

myst_heading_anchors = 2

autoapi_type = "python"
autoapi_dirs = [f"{Path(__file__).parents[2]}/"]

templates_path = ["_templates"]
exclude_patterns: Any = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
