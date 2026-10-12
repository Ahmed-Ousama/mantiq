import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mantiq")
except PackageNotFoundError:  # a source checkout that isn't installed
    __version__ = "0.0.0"


def __getattr__(name):
    # Import lazily so `import mantiq` stays light and `mantiq.cag` works without the optional loader dependencies.
    if name == "loaders":
        return importlib.import_module(".loaders", __name__)
    if name in ("NaiveCAG", "NaiiveCAG"):
        return getattr(importlib.import_module(".cag", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
