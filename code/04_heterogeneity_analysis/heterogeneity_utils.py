# heterogeneity_utils.py
# Forwards all imports to 00_heterogeneity_utils.py.
# Python can't import a module whose name starts with a digit using normal
# import syntax, so this shim lets callers write:
#   from heterogeneity_utils import run_heterogeneity
# while the actual source lives in the more readable 00_heterogeneity_utils.py.
import importlib, sys
sys.modules[__name__] = importlib.import_module("00_heterogeneity_utils")
