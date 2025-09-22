"""Jax utilities for zea."""

import jax


def str_to_jax_device(device):
    """Convert a device string to a JAX device.
    Args:
        device (str): Device string, e.g. ``'gpu:0'``, or ``'cpu:0'``.
    Returns:
        jax.Device: The corresponding JAX device.
    """

    if not isinstance(device, str):
        raise ValueError(f"Device must be a string, got {type(device)}")

    device = device.replace("cuda", "gpu")

    device = device.split(":")
    if len(device) == 2:
        device_type, device_number = device
        device_number = int(device_number)
    else:
        # if no device number is specified, use the first device
        device_type = device[0]
        device_number = 0

    if device_number > len(jax.devices(device_type)):
        raise ValueError(
            f"Device {device} is not available from JAX devices: {jax.devices(device_type)}"
        )

    return jax.devices(device_type)[device_number]
