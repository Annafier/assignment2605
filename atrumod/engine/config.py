"""Simple config loader — reads Python config files, returns dict-like object."""
import importlib.util
import sys
from pathlib import Path
from collections.abc import MutableMapping


class Config(MutableMapping):
    """Dict-like config with attribute access. Supports **unpacking."""

    def __init__(self, d=None):
        if d is None:
            d = {}
        object.__setattr__(self, '_data', {})
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            elif isinstance(v, list):
                v = [Config(x) if isinstance(x, dict) else x for x in v]
            self._data[k] = v

    def __getattr__(self, key):
        if key == '_data':
            return object.__getattribute__(self, '_data')
        if key in self._data:
            return self._data[key]
        raise AttributeError(f'Config has no key {key!r}')

    def __setattr__(self, key, value):
        if key == '_data':
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f'Config({self._data})'

    def get(self, key, default=None):
        return self._data.get(key, default)


def load_config(path):
    """Load a Python config file. Returns Config object."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)

    cfg_dict = {k: v for k, v in module.__dict__.items()
                if not k.startswith('_') and k.islower()}
    return Config(cfg_dict)
