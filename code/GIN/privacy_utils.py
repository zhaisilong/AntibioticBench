"""Helpers for shareable run metadata; never used for file access."""
from pathlib import PurePath, PureWindowsPath
import re

_PRIVATE_KEYS = {'argv', 'command', 'host', 'hostname', 'node', 'nodelist'}
_PATH_KEYS = ('path', 'file', 'csv', 'dir', 'root', 'checkpoint', 'executable')


def sanitize_metadata(value, key=''):
    """Remove identifying paths/commands while retaining scientific metadata."""
    name = key.lower()
    if name in _PRIVATE_KEYS:
        return None
    if isinstance(value, dict):
        return {k: sanitize_metadata(v, 'path' if name == 'paths' or name.endswith('_paths') else str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(v, key) for v in value]
    if isinstance(value, str):
        windows_abs = bool(re.match(r'^[A-Za-z]:[\\/]', value))
        if value.startswith('/') or windows_abs or any(name.endswith('_' + s) or name == s for s in _PATH_KEYS):
            if name == 'script':
                return PureWindowsPath(value).name if windows_abs else PurePath(value).name
            return '<configured>'
    return value


def install_privacy_excepthook():
    """Keep failure type and source locations without logging machine paths or values."""
    import sys
    import traceback

    def report(exc_type, exc, tb):
        print("Error in benchmark execution (details suppressed to avoid exposing local paths):", file=sys.stderr)
        for frame in traceback.extract_tb(tb):
            print(f"  {PurePath(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr)
        print(exc_type.__name__, file=sys.stderr)

    sys.excepthook = report
