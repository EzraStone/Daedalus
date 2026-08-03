"""A local web window onto the compiler.

Optional: needs ``pip install daedalus[web]``. The core pipeline — verifier,
compiler, corpus, evaluation — has no dependency on any of this.
"""

__all__ = ["app"]


def __getattr__(name: str):
    # Imported lazily so `import daedalus.web` does not hard-require FastAPI
    # merely to look the package up.
    if name == "app":
        from .app import app

        return app
    raise AttributeError(name)
