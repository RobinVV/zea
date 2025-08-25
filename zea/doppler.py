"""Doppler functions for processing I/Q ultrasound data."""

import keras
import numpy as np
from keras import ops

if keras.backend.backend() == "jax":
    import jax.numpy as jnp

    apply_along_axis = jnp.apply_along_axis
    correlate = ops.correlate
else:
    apply_along_axis = np.apply_along_axis
    correlate = np.correlate


def iq2doppler(
    data,
    center_frequency,
    pulse_repetition_frequency,
    sound_speed,
    hamming_size=None,
    lag=1,
):
    """Compute Doppler from packet of I/Q Data.

    Args:
        data (ndarray): I/Q complex data of shape (grid_size_z, grid_size_x, n_frames).
            n_frames corresponds to the ensemble length used to compute
            the Doppler signal.
        center_frequency (float): Center frequency of the ultrasound probe in Hz.
        pulse_repetition_frequency (float): Pulse repetition frequency in Hz.
        sound_speed (float): Speed of sound in the medium in m/s.
        hamming_size (int or tuple, optional): Size of the Hamming window to apply
            for spatial averaging. If None, no window is applied.
            If an integer, it is applied to both dimensions. If a tuple, it should
            contain two integers for the row and column dimensions.
        lag (int, optional): Lag for the auto-correlation computation.
            Defaults to 1, meaning Doppler is computed from the current frame
            and the next frame.

    Returns:
        doppler_velocities (ndarray): Doppler velocity map of shape (grid_size_z, grid_size_x) in
            meters/second.

    """
    assert data.ndim == 3, "Data must be a 3-D array"
    if not (isinstance(lag, int) and lag >= 1):
        raise ValueError("lag must be an integer >= 1")
    assert data.shape[-1] > lag, "Data must have more frames than the lag"

    if hamming_size is None:
        hamming_size = np.array([1, 1])
    elif np.isscalar(hamming_size):
        hamming_size = np.array([hamming_size, hamming_size])
    assert hamming_size.all() > 0 and np.all(hamming_size == np.round(hamming_size)), (
        "hamming_size must contain integers > 0"
    )

    # Auto-correlation method
    iq1 = data[:, :, : data.shape[-1] - lag]
    iq2 = data[:, :, lag:]
    autocorr = ops.sum(iq1 * ops.conj(iq2), axis=2)  # Ensemble auto-correlation

    # Spatial weighted average
    if hamming_size[0] != 1 and hamming_size[1] != 1:
        h_row = np.hamming(hamming_size[0])
        h_col = np.hamming(hamming_size[1])
        autocorr = apply_along_axis(lambda x: correlate(x, h_row, mode="same"), 0, autocorr)
        autocorr = apply_along_axis(lambda x: correlate(x, h_col, mode="same"), 1, autocorr)

    # Doppler velocity
    nyquist_velocities = sound_speed * pulse_repetition_frequency / (4 * center_frequency * lag)
    phase = ops.atan2(ops.imag(autocorr), ops.real(autocorr))
    doppler_velocities = -nyquist_velocities * phase / np.pi
    return doppler_velocities
