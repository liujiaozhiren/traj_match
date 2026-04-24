"""
`cursor_mod` contains patched/modernized entrypoints and utilities.

Some entrypoints shadow the repo-root `cfg` by doing:
  sys.modules["cfg"] = importlib.import_module("cursor_mod.cfg")
so the rest of the codebase can keep using `import cfg`.
"""

