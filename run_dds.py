import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import keras
from keras import ops

from zea import init_device, log
from zea.data import Dataset
from zea.internal.operators import FourierBlurOperator
from zea.models.diffusion import DiffusionModel
from zea.models.lpips import LPIPS
from zea.utils import translate
from zea.visualize import plot_image_grid, set_mpl_style

n_conditional_samples = 1
n_conditional_steps_dps = 100
n_conditional_steps_dds = 20

init_device(verbose=False)
set_mpl_style()

N_test_samples = 5

dataset = Dataset("/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonet_v2025/val/", key="data/image")

presets = list(DiffusionModel.presets.keys())
log.info(f"Available built-in zea presets for DiffusionModel: {presets}")

# Load LPIPS model for perceptual similarity
print("Loading LPIPS model...")
lpips_model = LPIPS.from_preset("lpips")

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

# Storage for all results
all_results = {
    'original': [],
    'blurred': [],
    'dds_reconstructed': [],
    'dps_reconstructed': [],
    'mse_blurred': [],
    'mse_dds': [],
    'mse_dps': [],
    'lpips_blurred': [],
    'lpips_dds': [],
    'lpips_dps': [],
    'time_dds': [],
    'time_dps': []
}

for i in range(N_test_samples):
    print(f"\n{'='*60}")
    print(f"Processing sample {i+1}/{N_test_samples}")
    print(f"{'='*60}")
    
    data = dataset[i]["data"]["image"][0] # grab just the first frame

    img_shape = model_dds.input_shape[:2]

    data = keras.ops.expand_dims(data, axis=-1)
    data = keras.ops.image.resize(data, img_shape)
    dynamic_range = (-60, 0)
    data = keras.ops.clip(data, dynamic_range[0], dynamic_range[1])
    data = translate(data, dynamic_range, (-1, 1))

    # Create the blurred measurement (this is our observed data y = Ax)
    # Add dummy batch dim
    blurred_measurement = blur_operator.forward(
        ops.expand_dims(data, axis=(0)),
    )

    measurement_input = blurred_measurement

    # Perform posterior sampling with DDS and measure time
    print(f"Performing posterior sampling with DDS guidance...")
    start_time_dds = time.time()
    posterior_samples_dds = model_dds.posterior_sample(
        measurements=measurement_input,
        n_samples=n_conditional_samples,
        n_steps=n_conditional_steps_dds,
        # verbose=True,
    )
    end_time_dds = time.time()
    dds_time = end_time_dds - start_time_dds

    # Perform posterior sampling with DPS and measure time
    print(f"Performing posterior sampling with DPS guidance...")
    start_time_dps = time.time()
    posterior_samples_dps = model_dps.posterior_sample(
        measurements=measurement_input,
        n_samples=n_conditional_samples,
        n_steps=n_conditional_steps_dps,
        omega=5.0,  # Guidance strength
        # verbose=True,
    )
    end_time_dps = time.time()
    dps_time = end_time_dps - start_time_dps

    # Extract results for DDS
    posterior_mean_dds = keras.ops.mean(posterior_samples_dds, axis=1)
    posterior_mean_dps = keras.ops.mean(posterior_samples_dps, axis=1)

    # Remove batch and channel dimensions for visualization
    posterior_mean_dds = keras.ops.squeeze(posterior_mean_dds)
    posterior_mean_dps = keras.ops.squeeze(posterior_mean_dps)

    # Calculate MSE metrics
    mse_measurement = keras.ops.mean((data - blurred_measurement) ** 2)
    mse_posterior_dds = keras.ops.mean((data - posterior_mean_dds) ** 2)
    mse_posterior_dps = keras.ops.mean((data - posterior_mean_dps) ** 2)

    # Prepare images for LPIPS (need to tile grayscale to RGB and add batch dimension)
    def prepare_for_lpips(img):
        img_squeezed = ops.squeeze(img)
        img_clipped = ops.clip(img_squeezed, -1, 1)
        img_tiled = ops.stack([img_clipped, img_clipped, img_clipped], axis=-1)
        return ops.expand_dims(img_tiled, axis=0)

    data_rgb = prepare_for_lpips(data)
    blurred_rgb = prepare_for_lpips(blurred_measurement)
    dds_rgb = prepare_for_lpips(posterior_mean_dds)
    dps_rgb = prepare_for_lpips(posterior_mean_dps)

    # Calculate LPIPS metrics
    print("Computing LPIPS scores...")
    lpips_blurred = lpips_model((data_rgb, blurred_rgb))[0]
    lpips_dds = lpips_model((data_rgb, dds_rgb))[0]
    lpips_dps = lpips_model((data_rgb, dps_rgb))[0]

    # Store results
    all_results['original'].append(ops.squeeze(data))
    all_results['blurred'].append(ops.squeeze(blurred_measurement))
    all_results['dds_reconstructed'].append(posterior_mean_dds)
    all_results['dps_reconstructed'].append(posterior_mean_dps)
    all_results['mse_blurred'].append(float(mse_measurement))
    all_results['mse_dds'].append(float(mse_posterior_dds))
    all_results['mse_dps'].append(float(mse_posterior_dps))
    all_results['lpips_blurred'].append(float(lpips_blurred))
    all_results['lpips_dds'].append(float(lpips_dds))
    all_results['lpips_dps'].append(float(lpips_dps))
    all_results['time_dds'].append(dds_time)
    all_results['time_dps'].append(dps_time)

# Create comprehensive comparison grid for all samples
print(f"\n{'='*60}")
print("Creating comprehensive results visualization...")
print(f"{'='*60}")

# Prepare images for grid (organize by rows: original, blurred, dds, dps)
grid_images = []
grid_titles = []

# Row 1: Original images
for i in range(N_test_samples):
    grid_images.append(all_results['original'][i])
    grid_titles.append(f"Original {i+1}")

# Row 2: Blurred images
for i in range(N_test_samples):
    grid_images.append(all_results['blurred'][i])
    grid_titles.append(f"Blurred {i+1}\nLPIPS: {all_results['lpips_blurred'][i]:.3f}")

# Row 3: DDS reconstructions
for i in range(N_test_samples):
    grid_images.append(all_results['dds_reconstructed'][i])
    grid_titles.append(f"DDS {i+1} ({all_results['time_dds'][i]:.1f}s)\nLPIPS: {all_results['lpips_dds'][i]:.3f}")

# Row 4: DPS reconstructions
for i in range(N_test_samples):
    grid_images.append(all_results['dps_reconstructed'][i])
    grid_titles.append(f"DPS {i+1} ({all_results['time_dps'][i]:.1f}s)\nLPIPS: {all_results['lpips_dps'][i]:.3f}")

# Create the comprehensive grid
fig_all, axes = plot_image_grid(
    grid_images,
    titles=grid_titles,
    vmin=-1,
    vmax=1,
    cmap="gray",
    ncols=N_test_samples,
    figsize=(3*N_test_samples, 12)
)
fig_all.savefig("dds_vs_dps_all_samples.png", dpi=150) #, bbox_inches="tight")
print(f"Saved all samples comparison to {log.yellow('dds_vs_dps_all_samples.png')}")

print(f"\nPer-Sample Detailed Results:")
print(f"{'Sample':<8}{'DDS Time':<10}{'DPS Time':<10}{'Speed Ratio':<12}{'MSE y':<10}{'MSE DDS':<10}{'MSE DPS':<10}{'LPIPS y':<10}{'LPIPS DDS':<12}{'LPIPS DPS':<12}")
print(f"{'-'*130}")
for i in range(N_test_samples):
    speed_ratio = all_results['time_dps'][i] / all_results['time_dds'][i]
    
    print(f"{i+1:<8}{all_results['time_dds'][i]:<10.2f}{all_results['time_dps'][i]:<10.2f}{speed_ratio:<12.2f}"
          f"{all_results['mse_blurred'][i]:<10.4f}{all_results['mse_dds'][i]:<10.4f}{all_results['mse_dps'][i]:<10.4f}"
          f"{all_results['lpips_blurred'][i]:<10.4f}{all_results['lpips_dds'][i]:<12.4f}{all_results['lpips_dps'][i]:<12.4f}")

print(f"\n{'='*80}")

del dataset