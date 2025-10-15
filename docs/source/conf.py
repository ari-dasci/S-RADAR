# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-informatio

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  



project = 'RADAR'
copyright = '2025, Beatriz Bello Garcia'
authors = ['Beatriz Bello García', 'Ignacio Aguilera Martos', 'Marina Hernández Bautista']
author = ', '.join(authors)
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              'sphinx.ext.autodoc',        # Documentación desde docstrings
              'sphinx.ext.napoleon',       # Estilo Google o NumPy en docstrings
              'sphinx.ext.viewcode',       # Añade enlaces al código fuente
              'myst_parser',               # Permite Markdown
             ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
