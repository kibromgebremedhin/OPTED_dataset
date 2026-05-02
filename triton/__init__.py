from . import language


class Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def jit(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


def autotune(*_args, **_kwargs):
    def decorator(fn):
        return fn

    return decorator


def cdiv(x, y):
    return (x + y - 1) // y
