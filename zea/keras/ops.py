"""This file creates a :class:`zea.Operation` for all unary :mod:`keras.ops`
and :mod:`keras.ops.image` functions.

They can be used in zea pipelines like any other :class:`zea.Operation`, for example:

.. code-block:: python

    from zea.keras import Squeeze

    op = Squeeze(axis=1)
"""

import inspect

import keras


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


def _generate_operation_class_code(name, func):
    """Generate Python code for a zea.Operation class for a given keras.ops function."""
    class_name = _snake_to_pascal(name)
    doc = f"Operation wrapping keras.ops.{name}."

    # Get the full module path for the function
    module_path = f"keras.ops.{name}" if hasattr(keras.ops, name) else f"keras.ops.image.{name}"

    return f'''
@ops_registry("{module_path}")
class {class_name}(Lambda):
    """{doc}"""
    
    def __init__(self, **kwargs):
        super().__init__(func={module_path}, **kwargs)

'''


def should_regenerate_ops_file(output_path="_generated_ops.py"):
    """Check if the generated ops file needs to be regenerated."""
    import os

    if not os.path.exists(output_path):
        return True

    # Check if Keras version in generated file matches current version
    try:
        with open(output_path, "r") as f:
            content = f.read()
            # Look for a version comment we'll add
            if f"# Generated for Keras {keras.__version__}" not in content:
                return True
    except Exception:
        return True

    return False


def generate_ops_file(output_path="_generated_ops.py"):
    """Generate a .py file with all operation class definitions."""
    _funcs = _unary_functions_from_namespace(keras.ops, "x")
    _funcs += _unary_functions_from_namespace(keras.ops.image, "images")

    # File header with version info
    content = f'''"""Auto-generated zea.Operations for all unary keras.ops functions.

This file is generated automatically. Do not edit manually.
# Generated for Keras {keras.__version__}
"""

import keras

from zea.internal.registry import ops_registry
from zea.ops import Lambda

'''

    # Generate all class definitions
    for name, func in _funcs:
        content += _generate_operation_class_code(name, func)

    # Write to file
    with open(output_path, "w") as f:
        f.write(content)


# Auto-regenerate if needed
if should_regenerate_ops_file("zea/keras/_generated_ops.py"):
    generate_ops_file("zea/keras/_generated_ops.py")

from ._generated_ops import *  # noqa: E402, F403
