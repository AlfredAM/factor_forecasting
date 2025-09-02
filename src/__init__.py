# src package init
import sys as _sys
import os as _os

# Ensure project root in sys.path for scripts installed via wheel
_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)