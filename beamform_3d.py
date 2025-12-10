import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive backend for matplotlib

import numpy as np
from keras import ops
from tqdm import tqdm

import zea
from zea import log
from zea.ops import (
    DelayAndSum,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Map,
    Normalize,
    Pipeline,
    ReshapeGrid,
    TOFCorrection,
)
from zea.tensor_ops import vmap
from zea.visualize import set_mpl_style

zea.init_device(verbose=False)
set_mpl_style()


def get_3d_grid(origin, vec0, vec1, vec2, shape, region_size, center):
    """
    Produces a 3D grid in 3D space.

    Args:
        origin (ops.array): The origin point of the grid of shape (3,).
        vec0 (ops.array): The first direction vector defining the grid of shape (3,).
        vec1 (ops.array): The second direction vector defining the grid of shape (3,).
        vec2 (ops.array): The third direction vector defining the grid of shape (3,).
        shape (tuple): The shape of the grid (number of points along each direction) of length 3.
        region_size (tuple): The size of the region covered by the grid of shape (3,).
        center (tuple): A tuple indicating whether to center the grid along each axis of length 3.

    Returns:
        grid (ops.array): The generated grid of shape (N, M, P, 3).
        v0_vals (ops.array): The distances along direction vec0 of shape (N,).
        v1_vals (ops.array): The distances along direction vec1 of shape (M,).
        v2_vals (ops.array): The distances along direction vec2 of shape (P,).
    """
    assert origin.shape == vec0.shape == vec1.shape == vec2.shape == (3,), (
        "origin, vec0, vec1, and vec2 must be of shape (3,)"
    )
    assert len(shape) == 3, "shape must be a tuple of length 3"
    assert len(region_size) == 3, "region_size must be a tuple of length 3"
    assert len(center) == 3, "center must be a tuple of length 3"

    # Center the origin along each axis if requested
    if center[0]:
        origin -= vec0 * region_size[0] / 2
    if center[1]:
        origin -= vec1 * region_size[1] / 2
    if center[2]:
        origin -= vec2 * region_size[2] / 2

    # Calculate voxel sizes
    delta = (
        region_size[0] / shape[0],
        region_size[1] / shape[1],
        region_size[2] / shape[2],
    )

    # Create coordinate arrays
    v0_vals = ops.arange(shape[0]) * delta[0]
    v1_vals = ops.arange(shape[1]) * delta[1]
    v2_vals = ops.arange(shape[2]) * delta[2]

    # Create 3D meshgrid
    v0_grid, v1_grid, v2_grid = ops.meshgrid(v0_vals, v1_vals, v2_vals, indexing="ij")

    # Construct the 3D grid
    grid = (
        v0_grid[:, :, :, None] * vec0[None, None, None, :]
        + v1_grid[:, :, :, None] * vec1[None, None, None, :]
        + v2_grid[:, :, :, None] * vec2[None, None, None, :]
    ) + origin

    return grid, v0_vals, v1_vals, v2_vals


def beamform_volume(data_iq, flatgrid, scan, probe, grid_shape):
    """Beamform a 3D volume for a given frame."""
    data_beamformed = None
    for n in range(data_iq.shape[1]):
        tx = ops.array([n])
        data_beamformed_new = zea.ops.tof_correction(
            data_iq[tx],
            flatgrid=flatgrid,
            t0_delays=scan.t0_delays[tx],
            tx_apodizations=scan.tx_apodizations[tx],
            sound_speed=scan.sound_speed,
            probe_geometry=probe.probe_geometry,
            initial_times=scan.initial_times[tx],
            sampling_frequency=scan.sampling_frequency,
            demodulation_frequency=scan.demodulation_frequency,
            tx_waveform_indices=scan.tx_waveform_indices[tx],
            t_peak=scan.t_peak,
            f_number=1.5,
            polar_angles=scan.polar_angles[tx],
            focus_distances=scan.focus_distances[tx],
        )
        if data_beamformed is None:
            data_beamformed = data_beamformed_new
        else:
            data_beamformed += data_beamformed_new

    # Post-process
    data_beamformed = ops.sum(data_beamformed, axis=(0, 2))
    data_beamformed = ops.linalg.norm(data_beamformed, axis=-1)
    data_beamformed = data_beamformed.reshape(grid_shape[0], grid_shape[1], grid_shape[2])
    data_beamformed = 20 * ops.log10(data_beamformed)
    data_beamformed = data_beamformed - data_beamformed.max()
    data_beamformed = ops.clip(data_beamformed, -40, 0)

    return data_beamformed


def beamform_4d_volume(data_iq, scan, probe, volume_shape, region_size, output_path):
    """
    Beamform the full 4D volume (X, Y, Z, T) using a 3D grid for each frame.

    Args:
        data_iq: IQ data of shape (n_frames, n_tx, ...)
        scan: Scan configuration
        probe: Probe configuration
        volume_shape: Tuple of (n_x, n_y, n_z) for the output volume
        region_size: Tuple of (x_size, y_size, z_size) in meters
        output_path: Path to save the NPZ file
    """
    n_frames = data_iq.shape[0]
    n_x, n_y, n_z = volume_shape
    x_size, y_size, z_size = region_size

    log.info(f"Beamforming 4D volume: {volume_shape} voxels, {n_frames} frames")
    log.info(f"Physical size: {x_size * 1e3:.1f} x {y_size * 1e3:.1f} x {z_size * 1e3:.1f} mm³")
    log.info(f"Total voxels per frame: {n_x * n_y * n_z:,}")

    # Initialize output volume: (n_frames, n_x, n_y, n_z)
    volume_4d = np.zeros((n_frames, n_x, n_y, n_z), dtype=np.float32)

    # Define orthogonal basis vectors
    vec_x = ops.array([1.0, 0.0, 0.0])
    vec_y = ops.array([0.0, 1.0, 0.0])
    vec_z = ops.array([0.0, 0.0, 1.0])

    # Create 3D grid once (same for all frames)
    origin = ops.array([0.0, 0.0, 0.0])
    log.info("Creating 3D grid...")
    grid, x_vals, y_vals, z_vals = get_3d_grid(
        origin=origin,
        vec0=vec_x,
        vec1=vec_y,
        vec2=vec_z,
        shape=volume_shape,
        region_size=region_size,
        center=(True, True, False),  # Center X and Y, not Z
    )

    # Flatten grid for beamforming
    flatgrid = grid.reshape(-1, 3)
    log.info(f"Grid shape: {grid.shape}, Flatgrid shape: {flatgrid.shape}")

    # Create coordinate arrays for saving
    x_coords = np.array(x_vals)
    y_coords = np.array(y_vals)
    z_coords = np.array(z_vals)

    # Beamform each frame
    for frame_idx in tqdm(range(n_frames), desc="Processing frames"):
        # Beamform entire 3D volume for this frame
        volume_data = beamform_volume(data_iq[frame_idx], flatgrid, scan, probe, volume_shape)

        # Store result
        volume_4d[frame_idx, :, :, :] = np.array(volume_data)

    # Save to NPZ file
    log.info(f"Saving beamformed volume to {log.yellow(output_path)}")
    np.savez_compressed(
        output_path,
        volume=volume_4d,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        shape=volume_shape,
        region_size=region_size,
    )
    np.save(output_path.replace(".npz", "_volume.npy"), volume_4d)

    log.info(f"✓ Saved 4D volume with shape {volume_4d.shape}")
    log.info(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    return volume_4d, x_coords, y_coords, z_coords


if __name__ == "__main__":
    # Load data
    file_path = "/mnt/z/Ultrasound-BMd/data/vincent/example-3d-data/carotid.hdf5"
    NUM_FRAMES = 3
    log.info(f"Loading data from {log.yellow(file_path)}")

    with zea.File(file_path, mode="r") as file:
        file.summary()
        rf_data_4d = file.load_data("raw_data", indices=list(range(NUM_FRAMES)))  # Load all frames
        scan = file.scan()
        probe = file.probe()

    scan.n_ch = 1

    pipeline = Pipeline(
        [
            Demodulate(),
            Map([TOFCorrection(), DelayAndSum()], argnames="flatgrid", chunks=128),
            ReshapeGrid(),
            EnvelopeDetect(),
            LogCompress(),
            Normalize(),
        ],
        with_batch_dim=True,
    )
    scan.grid_type = "cartesian"
    scan.grid_size_x = 128
    scan.grid_size_y = 128
    scan.grid_size_z = 128
    parameters = pipeline.prepare_parameters(probe, scan)

    def beamform_3d_volume(rf_data_3d):
        return pipeline(data=rf_data_3d, **parameters)["data"]

    n_frames = len(rf_data_4d)
    beamform_4d_volume = vmap(beamform_3d_volume, chunks=n_frames, in_axes=0)
    out = beamform_4d_volume(rf_data_4d[:, None, ...])

    out = pipeline(data=rf_data_4d, **parameters)

    # log.info("Demodulating data...")
    # data_iq = zea.ops.demodulate(
    #     data,
    #     center_frequency=scan.center_frequency,
    #     sampling_frequency=scan.sampling_frequency,
    # )

    # output_path = "/mnt/z/Ultrasound-BMd/data/oisin/carotid_mesh/beamformed_4d_debug.npz"

    # # Beamform and save
    # volume_4d, x_coords, y_coords, z_coords = beamform_4d_volume(
    #     data_iq=data_iq,
    #     scan=scan,
    #     probe=probe,
    #     volume_shape=volume_shape,
    #     region_size=region_size,
    #     output_path=output_path,
    # )

    # log.info("Done!")
    # log.info(f"Volume shape: {volume_4d.shape} (frames, x, y, z)")
    # log.info(f"Voxel size: {region_size[0]/volume_shape[0]*1e3:.2f} x {region_size[1]/volume_shape[1]*1e3:.2f} x {region_size[2]/volume_shape[2]*1e3:.2f} mm³")
