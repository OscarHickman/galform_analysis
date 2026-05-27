"""
Wrapper for the SCOPE correlation analysis tool.
"""

import os
import sys

# Ensure SCOPE's python bindings are in the path
_USER = os.environ.get("USER", "<USER>")
SCOPE_PATH = f"/cosma/apps/durham/{_USER}/SCOPE/python"
if SCOPE_PATH not in sys.path:
    sys.path.append(SCOPE_PATH)

try:
    import scope
except ImportError:
    scope = None


def is_scope_available():
    return scope is not None
