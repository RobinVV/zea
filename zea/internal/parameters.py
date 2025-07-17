"""
Parameter management system for ultrasound imaging.

This module provides the :class:`Parameters` base class, which implements
dependency-tracked, type-checked, and cacheable parameter logic for scientific
applications, primarily to support :class:`zea.Scan`.

See the Parameters class docstring for details on features and usage.
"""

import functools

import numpy as np

from zea.internal.cache import serialize_elements
from zea.internal.core import Object as ZeaObject
from zea.internal.core import _to_tensor


def cache_with_dependencies(*deps):
    """Decorator to mark a method as a computed property with dependencies."""

    def decorator(func):
        func._dependencies = deps

        @functools.wraps(func)
        def wrapper(self: Parameters):
            missing_set = self._missing_dependencies(func.__name__)
            if missing_set != set():
                raise MissingDependencyError(func.__name__, missing_set)

            if func.__name__ in self._cache:
                # Check if dependencies changed
                current_hash = self._current_dependency_hash(deps)
                if current_hash == self._dependency_versions.get(func.__name__):
                    return self._cache[func.__name__]

            result = func(self)
            self._computed.add(func.__name__)
            self._cache[func.__name__] = result
            self._dependency_versions[func.__name__] = self._current_dependency_hash(deps)
            return result

        return property(wrapper)

    return decorator


class MissingDependencyError(AttributeError):
    """Exception indicating that a dependency of an attribute was not met."""

    def __init__(self, attribute: str, missing_dependencies: set):
        super().__init__(
            f"Cannot access '{attribute}' due to missing dependencies: "
            + f"{sorted(missing_dependencies)}"
        )


class Parameters(ZeaObject):
    """Base class for parameters with dependencies.

    This class provides a robust parameter management system,
    supporting dependency tracking, lazy evaluation, and type validation.

    **Features:**

    - **Type Validation:** All parameters must be validated against their
      expected types as specified in the `VALID_PARAMS` dictionary.
      Setting a parameter to an invalid type raises a `TypeError`.

    - **Dependency Tracking:** Computed properties can declare dependencies on
      other parameters or properties using the `@cache_with_dependencies`
      decorator. The system automatically tracks and resolves these dependencies.

    - **Lazy Computation:** Computed properties are evaluated only when accessed,
      and their results are cached for efficiency.

    - **Cache Invalidation:** When a parameter changes, all dependent computed
      properties are invalidated and recomputed on next access.

    - **Leaf Parameter Enforcement:** Only leaf parameters
      (those directly listed in `VALID_PARAMS`) can be set. Attempting to set a computed
      property raises an informative `AttributeError` listing the leaf parameters
      that must be changed instead.

    - **Optional Dependency Parameters:** Parameters can be both set directly (as a leaf)
      or computed from dependencies if not set. If a parameter is present in `VALID_PARAMS`
      and also decorated with `@cache_with_dependencies`, it will use the explicitly set
      value if provided, or fall back to the computed value if not set or set to `None`.
      If you set such a parameter after it has been computed, the explicitly set value
      will override the computed value and remain in effect until you set it back to `None`,
      at which point it will again be computed from its dependencies. This pattern is useful
      for parameters that are usually derived from other values, but can also be overridden
      directly when needed, and thus don't have a forced relationship with the dependencies.

    - **Tensor Conversion:** The `to_tensor` method converts all parameters and optionally all
      computed properties to tensors for machine learning workflows.

    - **Error Reporting:** If a computed property cannot be resolved due to missing dependencies,
      an informative `AttributeError` is raised, listing the missing parameters.

    **Usage Example:**

    .. code-block:: python

        class MyParams(Parameters):
            VALID_PARAMS = {
                "a": {"type": int, "default": 1},
                "b": {"type": float, "default": 2.0},
                "d": {"type": float},  # optional dependency
            }

            @cache_with_dependencies("a", "b")
            def c(self):
                return self.a + self.b

            @cache_with_dependencies("a", "b")
            def d(self):
                if self._params.get("d") is not None:
                    return self._params["d"]
                return self.a * self.b


        p = MyParams(a=3)
        print(p.c)  # Computes and caches c
        print(p.c)  # Returns cached value

        # Changing a parameter invalidates the cache
        p.a = 4
        print(p.c)  # Recomputes c

        # You are not allowed to set computed properties
        # p.c = 5  # Raises AttributeError

        # Now check out the optional dependency, this can be either
        # set directly during initialization or computed from dependencies (default)
        print(p.d)  # Returns 6 (=3 * 2.0)
        p = MyParams(a=3, d=9.99)
        print(p.d)  # Returns 9.99

    """

    VALID_PARAMS = None

    def __init__(self, **kwargs):
        super().__init__()

        if self.VALID_PARAMS is None:
            raise NotImplementedError("VALID_PARAMS must be defined in subclasses of Parameters.")

        # Initialize parameters with defaults
        for param, config in self.VALID_PARAMS.items():
            if param not in kwargs and "default" in config:
                kwargs[param] = config["default"]

        # Validate parameter types
        for param, value in kwargs.items():
            if param not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {param}. "
                    f"Valid parameters are: {list(self.VALID_PARAMS.keys())}"
                )
            expected_type = self.VALID_PARAMS[param]["type"]
            if expected_type is not None and value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        allowed = ", ".join([t.__name__ for t in expected_type])
                        raise TypeError(
                            f"Parameter '{param}' expected type {allowed}, "
                            f"got {type(value).__name__}"
                        )
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' expected type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

        self._params = {}
        self._properties_with_dependencies, self._properties = self.get_properties()
        self._computed = set()
        self._cache = {}
        self._dependency_versions = {}
        for k, v in kwargs.items():
            self._params[k] = v

        # Tensor cache stores converted tensors for parameters and computed properties
        # to avoid converting them multiple times if there are no changes.
        self._tensor_cache = {}
        for name in self.__class__.__dict__:
            self._check_for_circular_dependencies(name)

    def __getattr__(self, item):
        # First check regular params
        if item in self._params:
            return self._params[item]

        # Then check if it's a known property on the class with dependencies
        cls_attr = getattr(type(self), item, None)
        if isinstance(cls_attr, property) and hasattr(cls_attr.fget, "_dependencies"):
            # Try to resolve dependencies
            missing_set = self._missing_dependencies(item)
            if missing_set == set():
                # Use descriptor protocol directly
                try:
                    return cls_attr.__get__(self, self.__class__)
                except Exception as e:
                    raise AttributeError(f"Error computing '{item}': {str(e)}")
            else:
                raise MissingDependencyError(item, missing_set)
        elif isinstance(cls_attr, property):
            # If it's a property without dependencies, just return it
            try:
                return cls_attr.__get__(self, self.__class__)
            except Exception as e:
                raise AttributeError(f"Error accessing property '{item}': {str(e)}")

        # Otherwise raise normal attribute error
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

    def __setattr__(self, key, value):
        # Give clear error message on assignment to methods
        class_attr = getattr(type(self), key, None)
        if callable(class_attr):
            raise AttributeError(
                f"Cannot assign to method '{key}'. "
                f"'{key}' is a method, not an attribute. "
                f"To use it, call it as a function, e.g.: '{self.__class__.__name__}.{key}(...)'"
            )

        if key.startswith("_"):
            super().__setattr__(key, value)
            return

        cls_attr = getattr(self.__class__, key, None)
        # Allow setting if it's a valid parameter, even if it's also a computed property
        if (
            isinstance(cls_attr, property)
            and hasattr(cls_attr.fget, "_dependencies")
            and key not in self.VALID_PARAMS
        ):
            # Only block if not a leaf parameter
            def find_leaf_params(name, seen=None):
                if seen is None:
                    seen = set()
                if name in seen:
                    return set()
                seen.add(name)
                attr = getattr(self.__class__, name, None)
                if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                    leaves = set()
                    for dep in attr.fget._dependencies:
                        leaves |= find_leaf_params(dep, seen)
                    return leaves
                else:
                    if name in self.VALID_PARAMS:
                        return {name}
                    return set()

            leaf_params = sorted(find_leaf_params(key))
            raise AttributeError(
                f"Cannot set computed property '{key}'. Only leaf parameters can be set. "
                f"To change '{key}', set one or more of its leaf parameters: {leaf_params}"
            )

        # Validate that parameter is in VALID_PARAMS
        if key not in self.VALID_PARAMS:
            raise ValueError(
                f"Invalid parameter: {key}. Valid parameters are: {list(self.VALID_PARAMS.keys())}"
            )

        # Validate parameter type
        expected_type = self.VALID_PARAMS[key]["type"]
        if expected_type is not None and value is not None:
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    allowed = ", ".join([t.__name__ for t in expected_type])
                    raise TypeError(
                        f"Parameter '{key}' expected type {allowed}, got {type(value).__name__}"
                    )
            else:
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Parameter '{key}' expected type {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # Set the parameter and invalidate dependencies
        self._params[key] = value

        # Invalidate cache for this parameter if it is also a computed property
        self._invalidate(key)

        self._invalidate_dependents(key)

    def _check_for_circular_dependencies(self, name, seen=None):
        """Check for circular dependencies in the dependency tree with a depth-first search."""
        if seen is None:
            seen = set()
        if name in seen:
            raise RuntimeError(f"Circular dependency detected for '{name}'")
        seen = seen.copy()
        seen.add(name)

        cls_attr = getattr(self.__class__, name, None)
        if isinstance(cls_attr, property) and hasattr(cls_attr.fget, "_dependencies"):
            for dep in cls_attr.fget._dependencies:
                self._check_for_circular_dependencies(dep, seen)

    def _find_all_dependents(self, target, seen=None):
        """
        Find all computed properties that depend (directly or indirectly) on the target parameter
        with a global search. Returns a set of property names that depend on the target.
        """
        dependents = set()
        if seen is None:
            seen = set()
        if target in seen:
            return dependents
        seen.add(target)
        for name in self.__class__.__dict__:
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                deps = attr.fget._dependencies
                if target in deps:
                    dependents.add(name)
                    # Recursively add dependents of this property
                    dependents |= self._find_all_dependents(name, seen)
        return dependents

    def _invalidate(self, key):
        """Invalidate a specific cached computed property and its dependencies."""
        self._cache.pop(key, None)
        self._computed.discard(key)
        self._dependency_versions.pop(key, None)
        self._tensor_cache.pop(key, None)
        self._invalidate_dependents(key)

    def _invalidate_dependents(self, changed_key):
        """
        Invalidate all cached computed properties that (directly or indirectly)
        depend on the changed_key.
        """
        for key in self._find_all_dependents(changed_key):
            self._invalidate(key)

    def _current_dependency_hash(self, deps) -> str:
        values = [self._params.get(dep, None) for dep in deps]
        return serialize_elements(values)

    def _missing_dependencies(self, name) -> set:
        missing_set = set()

        # Return immediately if already in params or cache
        if name in self._params or name in self._cache:
            return missing_set

        cls_attr = getattr(self.__class__, name, None)
        if isinstance(cls_attr, property):
            func = cls_attr.fget
            if hasattr(func, "_dependencies"):
                for dep in func._dependencies:
                    _missing_set = self._missing_dependencies(dep)
                missing_set.intersection_update(_missing_set)
        else:
            missing_set.add(name)

        return missing_set

    def get_properties(self):
        """
        Get all properties of this class
        """
        properties_with_dependencies = set()
        properties = set()
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, property):
                if hasattr(attr.fget, "_dependencies"):
                    properties_with_dependencies.add(name)
                else:
                    properties.add(name)
        return properties_with_dependencies, properties

    def to_tensor(self, include=None, exclude=None, keep_as_is: list = None):
        """
        Convert parameters and computed properties to tensors.

        Only one of `include` or `exclude` can be set.

        Args:
            include ("all", or list): Only include these parameter/property names.
                If "all", include all available parameters (i.e. their dependencies are met).
                Default is "all".
            exclude (None or list): Exclude these parameter/property names.
                If provided, these keys will be excluded from the output.
            keep_as_is (list): List of parameter/property names that should not be converted to
                tensors, but included as-is in the output.
        """
        if include is None and exclude is None:
            include = "all"

        if include is not None and exclude is not None:
            raise ValueError("Only one of 'include' or 'exclude' can be set.")

        # Determine which keys to include
        param_keys = set(self._params.keys())
        property_keys = set(self._properties_with_dependencies)
        all_keys = param_keys | property_keys | set(self._properties)

        if include is not None and include != "all":
            keys = set(include).intersection(all_keys)
        else:
            keys = set(all_keys)
        if exclude is not None:
            keys = keys - set(exclude)

        tensor_dict = {}
        # Convert parameters and computed properties to tensors
        for key in keys:
            # Get the value from params or computed properties
            try:
                val = getattr(self, key)
            except MissingDependencyError as exc:
                if include == "all":
                    # If we are including all, we can skip this key
                    continue
                else:
                    raise exc

            if key in self._tensor_cache:
                tensor_dict[key] = self._tensor_cache[key]
            else:
                tensor_val = _to_tensor(key, val, keep_as_is=keep_as_is)
                tensor_dict[key] = tensor_val
                self._tensor_cache[key] = tensor_val

        return tensor_dict

    def __repr__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"{k}=array(shape={v.shape})")
            else:
                param_lines.append(f"{k}={repr(v)}")

        param_str = ", ".join(param_lines)
        return f"{self.__class__.__name__}({param_str})"

    def __str__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"    {k}=array(shape={v.shape})")
            else:
                param_lines.append(f"    {k}={v}")

        param_str = ",\n".join(param_lines)
        return f"{self.__class__.__name__}(\n{param_str}\n)"

    @classmethod
    def safe_initialize(cls, **kwargs):
        """Overwrite safe initialize from zea.core.Object.

        We do not want safe initialization here.
        """
        return cls(**kwargs)
