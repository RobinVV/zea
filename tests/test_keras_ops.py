"""This tests the zea.keras module.

As we cannot test all functions, we will only test a few
of them to ensure that the wrapping works correctly."""

import numpy as np
import pytest

import zea.keras


def test_swapaxes():
    """Test the Swapaxes operation."""
    with pytest.raises(ValueError):
        zea.keras.Swapaxes(axis2=1)

    output = zea.keras.Swapaxes(axis1=0, axis2=1)(data=np.ones((10, 20)))["data"]
    assert output.shape == (20, 10)


def test_registry():
    """Test that all keras.ops functions are registered in ops_registry."""
    for name, _ in zea.keras._funcs:
        # Check if the operation class is created and registered
        op_class = getattr(zea.keras, zea.keras._snake_to_pascal(name), None)
        assert op_class is not None

        assert "keras." + name in zea.keras.ops_registry, (
            f"Operation {name} not registered in ops_registry"
        )
