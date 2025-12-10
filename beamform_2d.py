import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "1"

import matplotlib.pyplot as plt

from zea import init_device
from zea.data import load_file
from zea.display import to_8bit
from zea.ops import (
    DelayAndSum,
    EnvelopeDetect,
    LogCompress,
    Normalize,
    PatchedGrid,
    Pipeline,
    TOFCorrection,
)
from zea.visualize import set_mpl_style

init_device(verbose=False)
set_mpl_style()

path = "hf://zeahub/picmus/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"

data, scan, probe = load_file(
    path=path,
    indices=[0],
    data_type="raw_data",
)

# index the first frame
data_frame = data[0]

scan.n_ch = 2  # IQ data, should be stored in file but isn't currently
scan.xlims = probe.xlims
scan.zlims = (0, 0.06)  # reduce z-limits a bit for better visualization
dynamic_range = (-50, 0)  # set dynamic range for display

pipeline = Pipeline.from_default(
    num_patches=100,
    baseband=True,
    pfield=False,
    with_batch_dim=False,
    jit_options="pipeline",
)

parameters = pipeline.prepare_parameters(probe, scan)
parameters.pop("dynamic_range", None)  # remove dynamic_range since we will set it manually later

inputs = {pipeline.key: data_frame}

# dynamic parameters can be freely passed here as keyword arguments
outputs = pipeline(**inputs, **parameters)

image = outputs[pipeline.output_key]


def plot_data(data, dynamic_range, scan):
    """Helper function to plot the data."""
    image = to_8bit(data, dynamic_range=dynamic_range)
    plt.figure()
    # Convert xlims and zlims from meters to millimeters for display
    xlims_mm = [v * 1e3 for v in scan.xlims]
    zlims_mm = [v * 1e3 for v in scan.zlims]
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.imshow(image, cmap="gray", extent=[xlims_mm[0], xlims_mm[1], zlims_mm[1], zlims_mm[0]])
    plt.savefig("bf2d.png")


plot_data(image, dynamic_range, scan)
