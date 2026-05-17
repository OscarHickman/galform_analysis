"""
Wrapper for the SCOPE correlation analysis tool.
"""
import sys

# Ensure SCOPE's python bindings are in the path
SCOPE_PATH = "/cosma/apps/durham/dc-hick2/SCOPE/python"
if SCOPE_PATH not in sys.path:
    sys.path.append(SCOPE_PATH)

try:
    import scope
except ImportError:
    scope = None

def is_scope_available():
    return scope is not None
