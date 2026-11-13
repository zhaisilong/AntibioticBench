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
        return {k: sanitize_metadata(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(v, key) for v in value]
    if isinstance(value, str):
        windows_abs = bool(re.match(r'^[A-Za-z]:[\\/]', value))
        if value.startswith('/') or windows_abs or any(name.endswith('_' + s) or name == s for s in _PATH_KEYS):
            if name == 'script':
                return PureWindowsPath(value).name if windows_abs else PurePath(value).name
            return '<configured>'
    return value
