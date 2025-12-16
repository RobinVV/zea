import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive backend for matplotlib

import time

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

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
    # INPUT_PATH = "/mnt/z/Ultrasound-BMd/data/vincent/example-3d-data/carotid.hdf5"
    # INPUT_PATH = (
    #     "/mnt/z/Ultrasound-BMd/data/oisin/3D_acquisitions/Carotid/12_12_25_carotid_focused_3d.hdf5"
    # )
    INPUT_PATH = (
        "/mnt/z/Ultrasound-BMd/data/oisin/3D_acquisitions/CIRS/16_12_25_cirs_focused_3d.hdf5"
    )
    SAVE_PATH = "/mnt/z/Ultrasound-BMd/data/oisin/carotid_mesh/beamformed_3d_cirs.npy"
    NUM_FRAMES = 1
    log.info(f"Loading data from {log.yellow(INPUT_PATH)}")

    with zea.File(INPUT_PATH, mode="r") as file:
        file.summary()
        rf_data_4d = file.load_data("raw_data", indices=list(range(NUM_FRAMES)))
        scan = file.scan()
        probe = file.probe()

    scan.n_ch = 2
    # these params help a lot with contrast
    # scan.f_number = 1.5
    scan.dynamic_range = (-45, 0)

    pipeline = Pipeline(
        [
            Demodulate(),
            Map(
                [TOFCorrection(), DelayAndSum()],
                argnames="flatgrid",
                chunks=1024,  # INCREASE THIS IF YOU GET OOM
            ),
            ReshapeGrid(),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=True,
    )
    scan.grid_type = "cartesian"
    # Don't need to image the full depth
    scan.zlims = (0.0, 25e-3)
    # Decrease grid resolution from default for more efficient beamforming
    scan.grid_size_x = scan.grid_size_x // 2
    scan.grid_size_y = scan.grid_size_y // 2
    scan.grid_size_z = scan.grid_size_z // 2
    parameters = pipeline.prepare_parameters(probe, scan)

    def beamform_3d_volume(rf_data_3d):
        return pipeline(data=rf_data_3d[None, ...], **parameters)["data"][0]

    n_frames = len(rf_data_4d)
    beamform_4d_volume = vmap(beamform_3d_volume, chunks=n_frames, in_axes=0)

    start_time = time.time()
    beamformed_4d_volume = beamform_4d_volume(rf_data_4d)
    end_time = time.time()

    elapsed_time = end_time - start_time
    log.info(f"Beamforming completed in {log.yellow(f'{elapsed_time:.2f}')} seconds")
    log.info(f"Average time per frame: {log.yellow(f'{elapsed_time / NUM_FRAMES:.2f}')} seconds")
    np.save(SAVE_PATH, beamformed_4d_volume)
    log.info(f"Saved volume to: {log.yellow(f'{SAVE_PATH}')}")
