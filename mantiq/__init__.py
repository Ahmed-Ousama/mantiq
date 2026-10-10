import importlib


def __getattr__(name):
    # Import the loaders lazily so `mantiq.cag` works without the optional loader dependencies.
    if name == "loaders":
        return importlib.import_module(".loaders", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
