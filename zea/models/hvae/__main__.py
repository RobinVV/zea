import os

import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from keras import ops

from zea.models.hvae.hvae_model import VAE
from zea.models.hvae.hvae_utils import Parameters
from zea.models.generative import DeepGenerativeModel
from zea.internal.registry import model_registry
from zea.models.preset_utils import register_presets, get_preset_loader
from zea.models.presets import hvae_presets

SUPPORTED_VERSIONS = ["lvh", "lvh_ur24", "lvh_ur16", "lvh_ur8", "lvh_ur4"]

@model_registry(name="hvae")
class HVAE(DeepGenerativeModel):
    def __init__(
        self,
        name="hvae",
        version="lvh",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        assert version in SUPPORTED_VERSIONS, \
            f"Unsupported version '{version}' for HVAE model." \
            f"Current supported versions are: {', '.join(SUPPORTED_VERSIONS)}."
        self.version = version
        self.network = None

    def custom_load_weights(self, preset):
        loader = get_preset_loader(preset)
        args_file = loader.get_file("args.pkl")
        weights_file = loader.get_file(f"hvae_{self.version}.weights.h5")
        
        # Build the model architecture from args.pkl
        with open(args_file, "rb") as f:
            args = pickle.load(f)
        params = Parameters(args)
        
        vae = VAE(params)
        vae.build()
        
        # Load and copy the weights
        vae.load_weights(weights_file)
        vae.trainable = False
        self.network = vae
        
        # Set model parameters used in partial_inference
        self.depth = params.model_depth
        self.stage_depth = params.dec_num_blocks
        self.z_out = params.z_out

    def sample(self, n_samples=1, **kwargs):
        # Returns logits of shape (n_samples, 256, 256, 100)
        logits = self.network.decoder.call_uncond(n_samples, **kwargs)
        # Returns images of shape (n_samples, 256, 256, 3) in [-1, 1]
        samples = self.network.sample_from_mol(logits)
        return samples
        
    def posterior_sample(self, measurements, n_samples=1, **kwargs):
        # Measurements is [B, 256, 256, 3] in [-1, 1]
        B = ops.shape(measurements)[0]
        # Only need a single deterministic encoder pass
        activations = self.network.encoder(measurements)
        # Repeat the tensors in the list of activations n_samples amount of times
        # This repeats elementwise, so: [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        activations = [
            ops.repeat(a, repeats=n_samples, axis=0) for a in activations
        ]
        
        # Logits are of shape [B * n_samples, 256, 256, 100]
        logits, _, _ = self.network.decoder.call(activations)
        # Samples are of shape [B * n_samples, 256, 256, 3] in [-1, 1]
        samples = self.network.sample_from_mol(logits)
        
        # Split the samples into [B, n_samples, 256, 256, 3]
        output = ops.stack(ops.split(samples, B, axis=0), axis=0)
        return output
        
    def partial_inference(self, measurements, num_layers=0.5, num_images=1, **kwargs):
        """
        Performs TopDown inference with the HVAE up until a certain layer,
        after which it continues in the decoder with multiple prior streams.
        """
        # Make sure num_layers is a float between 0 and 1 or an integer between 1 and depth
        if isinstance(num_layers, float):
            assert 0.0 < num_layers <= 1.0, "num_layers as float must be in (0.0, 1.0]"
            num_layers = int(num_layers * self.depth)
        elif isinstance(num_layers, int):
            assert 1 <= num_layers <= self.depth, f"num_layers as int must be in [1, {self.depth}]"
        else:
            raise ValueError("num_layers must be either a float or an int.")

        B = ops.shape(measurements)[0]
        # Only need a single deterministic encoder pass
        activations = self.network.encoder(measurements)
        
        # Single pass through the top num_layers of the decoder
        # Adding the same latent to z_stage num_images times
        x = ops.zeros_like(activations[-1])
        z = ops.tile(ops.zeros([1, *self.z_out]), (B*num_images, 1, 1, 1))
        current_layer = 0
        for dec_stage, act in zip(self.network.decoder.stages.layers, reversed(activations)):
            for dec_block in dec_stage.blocks.layers:
                if current_layer < num_layers:
                    x, z_block, _ = dec_block.call(x, act)
                    z += ops.repeat(z_block, repeats=num_images, axis=0)
                else:
                    if current_layer == num_layers:
                        # If current_layer == num_layers, we duplicate the rest of the chain
                        x = ops.repeat(x, repeats=num_images, axis=0)
                    x, z_block = dec_block.call_uncond(x)
                    z += z_block
                current_layer += 1
            x = dec_stage.pool(x)
        
        z /= ops.sqrt(self.depth)
        px_z = self.network.decoder.activation(self.network.decoder.z_to_features(z))
        for out_block in self.network.decoder.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.network.decoder.last_conv(px_z)
        px_z = self.network.sample_from_mol(px_z)
        
        return ops.stack(ops.split(px_z, B, axis=0), axis=0)
    
    def log_density(self, data, **kwargs):
        # Returns the log_density in bits/dimension
        # Meaning:
        # KL loss of all latents summed, plus reconstruction loss per pixel summed
        # divided by (#pixels * ln(2))
        # as is standard for reporting VAE performance.
        recon, _, kl = self.network.call(data)
        # elbo is averaged over batch dimension
        elbo, _, _ = self.network.get_elbo(data, recon, kl, **kwargs)
        return elbo

register_presets(hvae_presets, HVAE)


if __name__ == "__main__":
    for ver in SUPPORTED_VERSIONS:
        model = HVAE.from_preset("hvae", version=ver)
    
    # from zea.agent.selection import UniformRandomLines
    # agent = UniformRandomLines(
    #                 n_actions = 24,
    #                 n_possible_actions = 256,
    #                 img_width = 256,
    #                 img_height = 256,
    #             )

        
    
    # model = HVAE.from_preset("hvae", version="lvh")
    # samples = model.sample(n_samples=3)
    # mask = ops.stack([agent.sample(batch_size=ops.shape(samples)[0])[1] for _ in range(3)], axis=-1)
    # measurements = ops.where(mask, samples, -1.0)
    
    # model.partial_inference(measurements=measurements, num_layers=0.5, num_images=4)
    
    
        
    


# TODO:
# Add function docstrings
    # Add support for docs
    # Add zeahub documentation
    # Add link to public swpenninga/hvae repo

# Reload final weights and save only VAE with trainable=False
# Upload new weights to huggingface

# Add tests
# Add example notebook

# Add function to do inference up until layer and return entropy plot + sample


