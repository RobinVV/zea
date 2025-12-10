import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive backend for matplotlib

import time

import numpy as np

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


if __name__ == "__main__":
    # Load data
    INPUT_PATH = "/mnt/z/Ultrasound-BMd/data/vincent/example-3d-data/carotid.hdf5"
    SAVE_PATH = "/mnt/z/Ultrasound-BMd/data/oisin/carotid_mesh/beamformed_4d_test.npy"
    NUM_FRAMES = 1
    log.info(f"Loading data from {log.yellow(INPUT_PATH)}")

    with zea.File(INPUT_PATH, mode="r") as file:
        file.summary()
        rf_data_4d = file.load_data("raw_data", indices=list(range(NUM_FRAMES)))  # Load all frames
        scan = file.scan()
        probe = file.probe()

    scan.n_ch = 2

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

    start_time = time.time()
    beamformed_4d_volume = beamform_4d_volume(rf_data_4d[:, None, ...])
    end_time = time.time()

    elapsed_time = end_time - start_time
    log.info(f"Beamforming completed in {log.yellow(f'{elapsed_time:.2f}')} seconds")
    log.info(f"Average time per frame: {log.yellow(f'{elapsed_time / NUM_FRAMES:.2f}')} seconds")
    np.save(SAVE_PATH, beamformed_4d_volume)
