import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import keras
from keras import ops

from zea import init_device, log
from zea.agent.selection import EquispacedLines
from zea.data import Dataset, File
from zea.internal.operators import FourierBlurOperator
from zea.models.diffusion import DiffusionModel
from zea.ops import Pipeline, ScanConvert
from zea.utils import translate
from zea.visualize import plot_image_grid, set_mpl_style

n_unconditional_samples = 16
n_unconditional_steps = 90
n_conditional_samples = 1
n_conditional_steps_dps = 100
n_conditional_steps_dds = 100

init_device(verbose=False)
set_mpl_style()

presets = list(DiffusionModel.presets.keys())
log.info(f"Available built-in zea presets for DiffusionModel: {presets}")

# Create blur operator with the same shape as your data
blur_operator = FourierBlurOperator(
    shape=(112, 112),
    cutoff_freq=0.1,  # Adjust this to control blur amount (0.1 = heavy blur, 0.8 = light blur)
    smooth=True,
)

# Create models with different guidance methods
model_dds = DiffusionModel.from_preset(
    "diffusion-echonet-dynamic", guidance="dds", operator=blur_operator
)
model_dps = DiffusionModel.from_preset(
    "diffusion-echonet-dynamic", guidance="dps", operator=blur_operator
)

pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = {
    "theta_range": [-0.78, 0.78],  # [-45, 45] in radians
    "rho_range": [0, 1],
}

with File(
    "/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonet_v2025/val/0X2267236AA0B2C189.hdf5"
) as file:
    validation_sample_frame = file.load_data("data/image", indices=0)

data = validation_sample_frame

img_shape = model_dds.input_shape[:2]

data = keras.ops.expand_dims(data, axis=-1)
data = keras.ops.image.resize(data, img_shape)
dynamic_range = (-60, 0)
data = keras.ops.clip(data, dynamic_range[0], dynamic_range[1])
data = translate(data, dynamic_range, (-1, 1))
data = data[..., 0]  # remove channel dim

# Create the blurred measurement (this is our observed data y = Ax)
blurred_measurement = blur_operator.forward(
    ops.expand_dims(data, axis=(0, -1)),
)

measurement_input = blurred_measurement

# Perform posterior sampling with DDS
print(f"\nPerforming posterior sampling with DDS guidance...")
posterior_samples_dds = model_dds.posterior_sample(
    measurements=measurement_input,
    n_samples=n_conditional_samples,
    n_steps=n_conditional_steps_dds,
    omega=5.0,  # Guidance strength
    verbose=True,
)

# Perform posterior sampling with DPS
print(f"\nPerforming posterior sampling with DPS guidance...")
posterior_samples_dps = model_dps.posterior_sample(
    measurements=measurement_input,
    n_samples=n_conditional_samples,
    n_steps=n_conditional_steps_dps,
    omega=5.0,  # Guidance strength
    verbose=True,
)

# Extract results for DDS
posterior_variance_dds = keras.ops.var(posterior_samples_dds, axis=1)
posterior_mean_dds = keras.ops.mean(posterior_samples_dds, axis=1)
posterior_sample_first_dds = posterior_samples_dds[:, 0]

# Extract results for DPS
posterior_variance_dps = keras.ops.var(posterior_samples_dps, axis=1)
posterior_mean_dps = keras.ops.mean(posterior_samples_dps, axis=1)
posterior_sample_first_dps = posterior_samples_dps[:, 0]

# Remove batch and channel dimensions for visualization
posterior_mean_dds = keras.ops.squeeze(posterior_mean_dds)
posterior_sample_first_dds = keras.ops.squeeze(posterior_sample_first_dds)

posterior_mean_dps = keras.ops.squeeze(posterior_mean_dps)
posterior_sample_first_dps = keras.ops.squeeze(posterior_sample_first_dps)

# Create comprehensive comparison plot
fig_comparison, axes = plot_image_grid(
    [
        ops.squeeze(data),
        ops.squeeze(blurred_measurement),
        posterior_mean_dds,
        posterior_mean_dps,
    ],
    titles=[
        r"$x^*$",
        r"$y = Ax$",
        f"DDS ({n_conditional_steps_dds} steps)",
        f"DPS ({n_conditional_steps_dps} steps)",
    ],
    vmin=-1,
    vmax=1,
    cmap="gray",
    ncols=4,
)
fig_comparison.savefig("dds_vs_dps_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved comparison to {log.yellow('dds_vs_dps_comparison.png')}")

# Print reconstruction metrics
mse_measurement = keras.ops.mean((data - blurred_measurement) ** 2)
mse_posterior_dds = keras.ops.mean((data - posterior_mean_dds) ** 2)
mse_posterior_dps = keras.ops.mean((data - posterior_mean_dps) ** 2)

print(f"MSE between original and blurred measurement: {mse_measurement:.4f}")
print(f"MSE between original and DDS posterior mean: {mse_posterior_dds:.4f}")
print(f"MSE between original and DPS posterior mean: {mse_posterior_dps:.4f}")
print(f"DDS improvement factor: {mse_measurement / mse_posterior_dds:.2f}x")
print(f"DPS improvement factor: {mse_measurement / mse_posterior_dps:.2f}x")
print(f"DDS vs DPS MSE ratio: {mse_posterior_dds / mse_posterior_dps:.3f}")