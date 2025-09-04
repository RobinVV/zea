"""This file creates zea.Operations for all unary keras.ops functions.

They can be used in zea pipelines like any other Operation, for example:

```python
from zea.keras import Squeeze

op = Squeeze(axis=1)
```
"""

import inspect

import keras

from zea.internal.registry import ops_registry
from zea.ops import Lambda


def _filter_funcs_by_first_arg(funcs, arg_name):
    """Filter a list of (name, func) tuples to those whose first argument matches arg_name."""
    filtered = []
    for name, func in funcs:
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params and params[0] == arg_name:
                filtered.append((name, func))
        except (ValueError, TypeError):
            # Skip functions that can't be inspected
            continue
    return filtered


def _functions_from_namespace(namespace):
    """Get all functions from a given namespace."""
    return [(name, obj) for name, obj in inspect.getmembers(namespace) if inspect.isfunction(obj)]


def _unary_functions_from_namespace(namespace, arg_name="x"):
    """Get all unary functions from a given namespace."""
    funcs = _functions_from_namespace(namespace)
    return _filter_funcs_by_first_arg(funcs, arg_name)


def _snake_to_pascal(name):
    """Convert a snake_case name to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _make_operation_class(name, func):
    """Create a zea.Operation class for a given keras.ops function."""
    class_name = _snake_to_pascal(name)
    doc = f"Operation wrapping keras.ops.{name}."

    def __init__(self, **kwargs):
        Lambda.__init__(self, func=func, **kwargs)

    return type(
        class_name,
        (Lambda,),
        {
            "__doc__": doc,
            "__init__": __init__,
            "call": Lambda.call,
        },
    )


_funcs = _unary_functions_from_namespace(keras.ops, "x")
_funcs += _unary_functions_from_namespace(keras.ops.image, "images")

for name, func in _funcs:
    cls = _make_operation_class(name, func)
    # Assign to module globals for direct access, e.g., zea.keras.Squeeze
    globals()[cls.__name__] = cls
    # Register the class in ops_registry using keras.ops name
    ops_registry("keras." + name)(cls)
