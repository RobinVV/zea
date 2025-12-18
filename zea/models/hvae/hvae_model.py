import random as pythonrandom
from keras import Model, Sequential, Variable, layers, ops, random, metrics
import numpy as np

from zea.models.hvae.hvae_utils import SoftPlus, GaussianAnalyticalKL, GradientNorms
from zea.agent.selection import UniformRandomLines


class VAETrainer(Model):
    """
    Trainer class for the HVAE, solely here for compatibility with loading weights.
    """

    def __init__(self, params, model=None, *args, **kwargs):
        super().__init__(**kwargs)

        self.params = params
        self.lr = params.learning_rate
        self.batch_size = params.batch_size

        if model is not None:
            self.model = model
            self.build_model = False
        else:
            self.model = VAE(params)
            self.build_model = True
        self.bits_dim_scale = ops.multiply(self.model.ndims, ops.log(2))

        self.agent = None
        if hasattr(params, "retrain_encoder"):
            if params.retrain_encoder:
                self.agent = UniformRandomLines(
                    n_actions=params.num_lines,
                    n_possible_actions=256,
                    img_width=256,
                    img_height=256,
                )

        self.elbo_tracker = metrics.Mean(name="loss")
        self.recon_tracker = metrics.Mean(name="recon")
        self.kl_tracker = metrics.Mean(name="kl")
        self.grad_tracker = GradientNorms(name="grad_norm")
        self.skip_tracker = metrics.Sum(name="skips")
        self.grad_skipnorm = Variable(1e9, trainable=False, dtype="float32")

    def build(self):
        shape_tuple = (
            None,  # batch dimension can be None
            self.params.enc_input_size[0],  # height
            self.params.enc_input_size[0],  # width
            self.params.data_width,  # channels
        )
        super().build(shape_tuple)
        if self.build_model:
            self.model.build()


class VAE(Model):
    """
    Hierarchical Variational Autoencoder (HVAE) model class.
    Contains an encoder, decoder, and methods for ELBO computation,
    sampling, and model printing.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.beta_max = params.beta
        self.beta_warmup_epochs = params.beta_warmup_epochs

        beta_start = 0 if self.beta_warmup_epochs > 0 else params.beta
        self.beta = Variable(beta_start, trainable=False, dtype="float32")

        self.loss_fn = params.loss_fn
        self.ndims = params.enc_input_size[0] ** 2 * params.data_width
        self.gradient_smoothing = params.gradient_smoothing
        self.min_mol_logscale = -250
        self.softplus = SoftPlus(0.69314718056)

    def build(self):
        self.encoder.build()
        self.decoder.build()
        _ = self.call(
            random.normal(
                [
                    2,  # Batch_size is at least 2 in case of cr-vae
                    self.params.enc_input_size[0],
                    self.params.enc_input_size[0],
                    self.params.data_width,
                ]
            )
        )
        _ = self.call(
            random.normal(
                [
                    2,  # Batch_size is at least 2 in case of cr-vae
                    self.params.enc_input_size[0],
                    self.params.enc_input_size[0],
                    self.params.data_width,
                ]
            ),
        )

    def get_elbo(self, x, px_z, kl, mask=None):
        """
        Calculates the ELBO given the input data, reconstructed output,
        and KL-divergences originating from the model.
        """
        recon_total = self.loss_fn.call(targets=x, logits=px_z, mask=mask)
        kl_total = ops.zeros(ops.shape(x)[0])

        for kls_stage in kl:
            for kl_block in kls_stage:
                if isinstance(kl_block, (list, tuple)) and len(kl_block) == 2:
                    kl_total += ops.sum(kl_block[0], axis=[1, 2, 3])
                    kl_total -= kl_block[1]
                else:
                    kl_total += ops.sum(kl_block, axis=[1, 2, 3])

        if self.params.channels_out > 1:
            recon_total /= self.ndims * ops.log(2)
            kl_total /= self.ndims * ops.log(2)

        recon_total = ops.mean(recon_total)
        kl_total = ops.mean(kl_total)
        elbo = recon_total + self.beta * kl_total

        return elbo, recon_total, kl_total

    def call(self, x):
        """
        Performs a forward pass through the encoder and decoder,
        returning the reconstructed output, latent samples, and KL divergences.
        """
        activations = self.encoder(x)
        px_z, z, kl = self.decoder(activations)
        return px_z, z, kl

    def sample_from_mol(self, logits, t=1.0):
        """
        Samples from a mixture of logistics parameterized by the logits.
        Generally, converts the 100-channel mixture into a 3-frame image.
        """

        # Same thing as in the DiscMixLogisticLoss
        B, H, W, _ = ops.shape(logits)

        # Unpack parameters
        logit_probs = logits[:, :, :, : self.params.num_output_mixtures]  # B, H, W, M
        l = logits[:, :, :, self.params.num_output_mixtures :]  # B, H, W, M*C*3
        l = layers.Reshape([H, W, self.params.data_width, 3 * self.params.num_output_mixtures])(
            l
        )  # B, H, W, C, 3*M

        # sample mixture indicator from softmax
        gumbel_noise = -ops.log(
            -ops.log(
                random.uniform(
                    (B, H, W, self.params.num_output_mixtures), minval=1e-5, maxval=1.0 - 1e-5
                )
            )
        )  # B, H, W, M
        amax = ops.argmax(logit_probs / t + gumbel_noise, axis=-1)  # B, H, W
        lambda_ = ops.cast(
            ops.one_hot(amax, self.params.num_output_mixtures, axis=-1), "float32"
        )  # B, H, W, M
        lambda_ = ops.expand_dims(lambda_, axis=3)  # B, H, W, 1, M

        means = ops.sum(
            l[:, :, :, :, : self.params.num_output_mixtures] * lambda_, axis=-1
        )  # B, H, W, C

        log_scales = self.min_mol_logscale + self.softplus(
            l[:, :, :, :, self.params.num_output_mixtures : 2 * self.params.num_output_mixtures]
            - self.min_mol_logscale
        )  # B, H, W, C
        log_scales = ops.sum(log_scales * lambda_, axis=-1)  # B, H, W, C

        coeffs = ops.sum(
            ops.tanh(
                l[
                    :,
                    :,
                    :,
                    :,
                    2 * self.params.num_output_mixtures : 3 * self.params.num_output_mixtures,
                ]
            )
            * lambda_,
            axis=-1,
        )  # B, H, W, C

        # Sample from logistic
        u = random.uniform((B, H, W, self.params.data_width), minval=1e-5, maxval=1.0 - 1e-5)
        x = means + ops.exp(log_scales) * t * (ops.log(u) - ops.log(1.0 - u))  # B, H, W, C

        # Auto-regressive sampling RGB and clip
        if self.params.data_width == 3:
            x0 = ops.clip(x[:, :, :, 0:1], -1, 1)
            x1 = ops.clip(x[:, :, :, 1:2] + coeffs[:, :, :, 0:1] * x0, -1, 1)
            x2 = ops.clip(
                x[:, :, :, 2:3] + coeffs[:, :, :, 1:2] * x0 + coeffs[:, :, :, 2:3] * x1,
                -1,
                1,
            )
            x = ops.concatenate([x0, x1, x2], axis=-1)  # B, H, W, C
        else:
            x = ops.clip(x, -1, 1)

        return x

    def print_model(self):
        """
        Prints the architecture and parameter counts of the HVAE model.
        """
        print("------ Encoder ------")
        print(
            (
                f"first 1x1 conv: [None, {self.params.enc_input_size[0], self.params.enc_input_size[0], self.params.data_width}]"
                f" -> [None, {self.params.enc_input_size[0], self.params.enc_input_size[0], self.encoder.first_conv.weights[0].shape[-1]}]"
            )
        )
        for stage_num, stage in enumerate(self.encoder.stages.layers):
            kernel_size = self.params.kernelsizes[str(stage.input_size)]
            stage_params = 0
            for block in stage.blocks.layers:
                stage_params += block.count_params()
            stage_params += stage.pool.count_params()
            print(
                (
                    f"stage:{stage_num:2} - input_size:{stage.input_size:3} - in_width:{stage.in_width:3}"
                    f" - middle_width:{stage.middle_width:3} - #blocks:{stage.num_blocks:2} - ksize:{kernel_size:2} - #params:{stage_params:,d}"
                )
            )
        print("------ Decoder ------")
        for stage_num, stage in enumerate(self.decoder.stages.layers):
            stage_params = 0
            for block in stage.blocks.layers:
                if self.params.use_depthwise_attention:
                    stage_params += block.queries.count_params()
                    if block.combine_queries:
                        stage_params += block.queries_comb_q.count_params()
                        if not block.first_block:
                            stage_params += block.queries_comb_p.count_params()
                stage_params += block.q.count_params()

                stage_params += block.p.count_params()
                stage_params += block.z_out_f.count_params()
                if not block.last_block:
                    stage_params += block.res.count_params()
                    stage_params += block.z_proj.count_params()
            stage_params += stage.pool.count_params()

            print(
                (
                    f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - in_width:{stage.in_width:3}"
                    f" - middle_width:{stage.middle_width:3} - #layers:{stage.num_blocks:2} - z_ch:{stage.zdim:3} - #params:{stage_params:,d}"
                )
            )
        output_blocks_params = 0
        for block in self.decoder.output_blocks.layers:
            output_blocks_params += block.count_params()
        print(
            (
                f"--- Output ---\n"
                f"blocks   - input_size:{self.decoder.output_shape[1]:3} - in_width:{self.params.z_out_width:3} - middle_width:{self.params.z_out_middle_width:3} - #blocks:{self.decoder.num_output_blocks:2} - ksize:{self.params.kernelsizes[str(self.params.dec_input_size[-1])]:2} - #params:{output_blocks_params:,d}\n"
                f"last 3x3 conv: [None, {stage.input_size, stage.input_size, self.params.z_out_width}]"
                f" -> [None, {stage.input_size, stage.input_size, self.decoder.last_conv.weights[0].shape[-1]}]"
            )
        )
        print("------  Flows  ------")
        if self.params.flow_type == "none":
            print(f"        {None}")
        else:
            for stage_num, stage in enumerate(self.decoder.stages.layers):
                stage_params = 0
                for block in stage.blocks.layers:
                    if block.use_flow:
                        stage_params += block.flows.count_params()
                if self.params.flow_type == "sylvester":
                    print(
                        f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - eff_zdim:{stage.zdim * stage.input_size * stage.input_size:5} - #flows:{stage.num_flows:2} - num_ortho_vecs:{stage.num_ortho_vecs:2} - flow_in_ch:{stage.flow_in_ch:2} - #params:{stage_params:,d}"
                    )
                else:
                    channels = 0
                    flows_per_level = 0
                    split_first = False
                    n_levels = 0
                    use_flow = False
                    if len(stage.blocks.layers) > 0:
                        if stage.blocks.layers[0].use_flow:
                            channels = stage.blocks.layers[0].flows.sylv_channels
                            flows_per_level = stage.blocks.layers[0].flows.flows_per_level
                            split_first = stage.blocks.layers[0].flows.split_first
                            n_levels = stage.blocks.layers[0].flows.n_levels
                            use_flow = True
                    print(
                        f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - eff_zdim:{stage.zdim * stage.input_size * stage.input_size:5} - flow:{use_flow:1} - width: {channels:3} - #flows_per_level:{flows_per_level} - splitfirst: {split_first:1} - n_levels: {n_levels:2} - #params:{stage_params:,d}"
                    )
            print(f"flow_type:      {self.params.flow_type}")
            if self.params.flow_type == "conv_sylvester":
                print(f"spectral_norm:  {self.params.spectral_norm}")

        print("---- Attention parameters ----")
        print(f"spatial_attention:   {self.params.use_spatial_attention}")
        if self.params.use_spatial_attention:
            print(f"s_a_width:           {self.params.enc_sa_width}")
        print(f"depthwise_attention: {self.params.use_depthwise_attention}")
        if self.params.use_depthwise_attention:
            print(f"query_width:         {self.params.query_width}")
            print(f"num_queries:         {self.params.num_queries}\n")

        print("---- Model settings ----")
        print(
            (
                f"init_zeros:     {self.params.init_zeros}\n"
                f"block_act:      {self.params.b_act_name}\n"
                f"pool_act:       {self.params.p_act_name}\n"
                f"group_norm:     {self.params.block_bn}\n"
                f"depthwise_conv: {self.params.depthwise}\n"
                f"model_depth:    {self.params.model_depth}\n"
                f"num_enc_blocks: {sum(self.params.enc_num_blocks)}\n"
                f"num_output_mix: {self.params.num_output_mixtures}\n"
                f"grad_smoothing: {self.params.gradient_smoothing}"
            )
        )


class Decoder(layers.Layer):
    """
    Decoder part of the Hierarchical Variational Autoencoder (HVAE).
    Contains the stages that are separated by resolution levels
    (powers of 2).
    """

    def __init__(self, params):
        super().__init__()
        self.output_shape = [
            1,
            params.dec_input_size[-1],
            params.dec_input_size[-1],
            params.dec_in_width[-1],
        ]
        self.z_features_shape = (
            1,
            params.z_out[0],
            params.z_out[1],
            params.z_out[2],
        )
        self.num_stages = params.num_stages
        self.z_to_features = layers.Conv2D(params.z_out_width, kernel_size=1, padding="same")
        self.model_depth = params.model_depth
        self.num_output_blocks = params.output_blocks
        self.init_shape = [
            params.dec_input_size[0],
            params.dec_input_size[0],
            params.dec_in_width[0],
        ]
        self.init_zeros = params.init_zeros
        if not self.init_zeros:
            self.init_bias = self.add_weight(
                shape=(1, self.init_shape[0], self.init_shape[1], self.init_shape[2]),
                initializer="zeros",
                trainable=True,
            )

        self.use_depthwise_attention = params.use_depthwise_attention

        self.stages = Sequential()
        for num in range(self.num_stages):
            self.stages.add(DecoderStage(params, num))

        self.activation = params.block_activation
        self.output_blocks = Sequential()
        for num in range(self.num_output_blocks):
            self.output_blocks.add(
                Block(
                    input_size=self.output_shape[1],
                    in_width=params.z_out_width,
                    middle_width=params.z_out_middle_width,
                    out_width=params.z_out_width,
                    kernelsize=params.kernelsizes[str(params.dec_input_size[-1])],
                    activation=params.block_activation,
                    bn=params.block_bn,
                    residual=True,
                    zero_last=False,
                    model_depth=params.model_depth,
                    depthwise=params.depthwise,
                    use_attention=params.use_spatial_attention,
                    attention_width=params.dec_sa_width[-1],
                )
            )
        self.last_conv = layers.Conv2D(
            params.channels_out,
            kernel_size=params.kernelsizes[str(params.dec_input_size[-1])],
            padding="same",
        )

    def build(self):
        for dec_stage in self.stages.layers:
            dec_stage.build()
        for out_block in self.output_blocks.layers:
            out_block.build()

    def depthwise_attention_call(self, activations):
        """
        Depthwise attention is not implemented in this code snippet.
        """
        z_stages = []
        kl_stages = []

        if self.init_zeros:
            x = ops.zeros(
                shape=(
                    ops.shape(activations[0][0])[0],
                    self.init_shape[0],
                    self.init_shape[1],
                    self.init_shape[2],
                )
            )
        else:
            b = ops.shape(activations[0][0])[0]
            x = ops.repeat(self.init_bias, b, axis=0)

        # vp and kp get overwritten in first call, theyre placeholders
        vp = ops.zeros(1)
        kp = ops.zeros(1)

        # pack top-down stream as x, vp, kp
        x = (x, vp, kp)

        # We give multiple activations to each decoder stage, to have
        # v^q_>l and k^q_>l
        # activations are 32x32, 16x16, 8x8, etc.
        # we convert a list with different resolutions to a
        # list containing tensors of same resolution with dimension (layers)
        activations = prepare_activations(activations)

        for dec_stage, act in zip(self.stages.layers, reversed(activations)):
            # First stage gets [32x32, 16x16, ..., 1x1], last stage gets [32x32]
            x, z, kl = dec_stage(x, act)
            z_stages += [z]
            kl_stages += [kl]
        return x, z_stages, kl_stages

    def call(self, activations):
        """
        Creates an output image from the encoder activations.
        args:
            activations: List of encoder activations at different resolutions.
        returns:
            Reconstructed image, latent samples, KL divergences.

        - Starts from an empty tensor or learned bias.
        - Passes through decoder stages with encoder activations.
        - Sums all latents.
        - Passes this latent through output blocks and reconstructs image.

        """
        if self.use_depthwise_attention:
            x, z_stages, kl_stages = self.depthwise_attention_call(activations)
        else:
            z_stages = []
            kl_stages = []

            if self.init_zeros:
                x = ops.zeros_like(activations[-1])
            else:
                b = ops.shape(activations[-1])[0]
                x = ops.repeat(self.init_bias, b, axis=0)

            for dec_stage, act in zip(self.stages.layers, reversed(activations)):
                x, z, kl = dec_stage(x, act)
                z_stages.append(z)
                kl_stages.append(kl)

        z_out = sum(z_stages) / ops.sqrt(self.model_depth)

        px_z = self.activation(self.z_to_features(z_out))
        for out_block in self.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.last_conv(px_z)

        return px_z, z_stages, kl_stages

    def call_uncond(self, num_images=16, t=1):
        """
        Generates an image from the prior of every stage.
        """
        if self.init_zeros:
            x = ops.zeros((num_images, self.init_shape[0], self.init_shape[1], self.init_shape[2]))
        else:
            x = ops.repeat(self.init_bias, num_images, axis=0)

        if self.use_depthwise_attention:
            # vp and kp get overwritten in first call, these are placeholders
            vp = ops.zeros(1)
            kp = ops.zeros(1)
            # pack top-down stream as x, vp, kp
            x = (x, vp, kp)

        z_stages = []
        for dec_stage in self.stages.layers:
            x, z = dec_stage.call_uncond(x, t)
            z_stages.append(z)

        z_out = sum(z_stages) / ops.sqrt(self.model_depth)

        px_z = self.activation(self.z_to_features(z_out))
        for out_block in self.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.last_conv(px_z)

        return px_z


class Encoder(layers.Layer):
    """
    Encoder part of the Hierarchical Variational Autoencoder (HVAE).
    Contains the stages that are separated by resolution levels.
    """

    def __init__(self, params):
        super().__init__()
        self.input_shape = [
            1,
            params.enc_input_size[0],
            params.enc_input_size[0],
            params.data_width,
        ]
        self.num_stages = params.num_stages
        self.first_conv = layers.Conv2D(
            params.enc_in_width[0],
            kernel_size=params.kernelsizes[str(params.enc_input_size[0])],
            padding="same",
        )

        self.stages = Sequential()
        for num in range(self.num_stages):
            self.stages.add(EncoderStage(params, num))

    def build(self):
        for enc_stage in self.stages.layers:
            enc_stage.build()
        _ = self.call(random.normal(self.input_shape))

    def call(self, x):
        """
        Args:
            x: Input tensor to the encoder.
        Returns:
            List of encoder activations at different resolutions.
        """
        x = self.first_conv(x)
        activations = []
        for enc_stage in self.stages.layers:
            # If there is no depth_wise attention, activations are passed to decoder
            # If there is depth_wise attention, act[0]=v^q, act[1]=k^q
            # every v^q and k^q is at a different resolution,
            # pooling/upsampling happens at attention block

            x, act = enc_stage(x)
            activations += [act]
        return activations


class DecoderStage(layers.Layer):
    def __init__(self, params, stage_num):
        super().__init__()
        self.zdim = params.zdim[stage_num]
        self.input_size = params.dec_input_size[stage_num]
        self.in_width = params.dec_in_width[stage_num]
        self.middle_width = params.dec_middle_width[stage_num]
        self.pool_width = params.dec_pool_width[stage_num]
        self.num_blocks = params.dec_num_blocks[stage_num]

        # attention
        self.use_spatial_attention = params.use_spatial_attention
        self.sa_width = params.dec_sa_width[stage_num]
        self.use_depthwise_attention = params.use_depthwise_attention

        # sylvester flow params
        self.num_flows = params.num_flows[stage_num]
        self.flow_in_ch = params.flow_in_ch[stage_num] if self.num_flows > 0 else 0
        self.num_ortho_vecs = params.num_ortho_vecs[stage_num] if self.num_flows > 0 else 0

        # conv_sylvester flow param
        self.convsylv_channels = params.convsylv_channels[stage_num]
        self.convsylv_flows_per_stage = params.convsylv_flows_per_stage[stage_num]
        self.convsylv_splitfirst = params.convsylv_splitfirst[stage_num]
        self.convsylv_stage_limit = params.convsylv_stage_limit[stage_num]

        self.z_out = params.z_out

        # For the decoder, a block is a DecBlock
        self.blocks = Sequential()
        for i in range(self.num_blocks):
            first_block = False
            last_block = False
            if i == 0 and self.input_size == min(params.dec_input_size):
                first_block = True
            if i == self.num_blocks - 1 and self.input_size == max(params.dec_input_size):
                last_block = True

            self.blocks.add(
                DecBlock(
                    input_size=self.input_size,
                    in_width=self.in_width,
                    middle_width=self.middle_width,
                    zdim=self.zdim,
                    z_out=params.z_out,
                    kernelsize=params.kernelsizes[str(self.input_size)],
                    block_activation=params.block_activation,
                    block_bn=params.block_bn,
                    model_depth=params.model_depth,
                    depthwise=params.depthwise,
                    num_flows=self.num_flows,
                    flow_in_ch=self.flow_in_ch,
                    num_ortho_vecs=self.num_ortho_vecs,
                    flow_type=params.flow_type,
                    spectral_norm=params.spectral_norm,
                    convsylv_channels=self.convsylv_channels,
                    convsylv_flows_per_stage=self.convsylv_flows_per_stage,
                    convsylv_splitfirst=self.convsylv_splitfirst,
                    convsylv_stage_limit=self.convsylv_stage_limit,
                    gradient_smoothing=params.gradient_smoothing,
                    use_spatial_attention=self.use_spatial_attention,
                    sa_width=self.sa_width,
                    use_depthwise_attention=self.use_depthwise_attention,
                    query_width=params.query_width,
                    num_queries=params.num_queries,
                    first_block=first_block,
                    last_block=last_block,
                )
            )
        self.pool = PoolLayer(
            input_size=self.input_size,
            in_width=self.in_width,
            out_width=self.pool_width,
            pool_activation=params.pool_activation,
            unpool=True,
            data_size=params.enc_input_size[0],
        )
        if self.use_depthwise_attention:
            self.attn_pool = AttentionUnpool(
                input_size=self.input_size,
                data_size=params.enc_input_size[0],
            )

    def build(self):
        for dec_block in self.blocks.layers:
            dec_block.build()
        self.pool.build()

    def call(self, x, act):
        B = ops.shape(x[0])[0] if self.use_depthwise_attention else ops.shape(x)[0]
        z_blocks = ops.tile(ops.zeros([1, *self.z_out]), (B, 1, 1, 1))
        kl_blocks = []
        for dec_block in self.blocks.layers:
            x, z, kl = dec_block.call(x, act)
            z_blocks += z
            kl_blocks.append(kl)

        if not self.use_depthwise_attention:
            x = self.pool(x)
        else:
            dec, vp, kp = x
            x = (self.pool(dec), self.attn_pool(vp), self.attn_pool(kp))
        return x, z_blocks, kl_blocks

    def call_uncond(self, x, t=1):
        B = ops.shape(x[0])[0] if self.use_depthwise_attention else ops.shape(x)[0]
        z_blocks = ops.tile(ops.zeros([1, *self.z_out]), (B, 1, 1, 1))
        for dec_block in self.blocks.layers:
            x, z = dec_block.call_uncond(x, t)
            z_blocks += z

        if not self.use_depthwise_attention:
            x = self.pool(x)
        else:
            dec, vp, kp = x
            x = (self.pool(dec), self.attn_pool(vp), self.attn_pool(kp))
        return x, z_blocks


class EncoderStage(layers.Layer):
    def __init__(self, params, stage_num):
        super().__init__()
        self.stage_num = stage_num

        self.input_size = params.enc_input_size[stage_num]
        self.in_width = params.enc_in_width[stage_num]
        self.middle_width = params.enc_middle_width[stage_num]
        self.pool_width = params.enc_pool_width[stage_num]
        self.num_blocks = params.enc_num_blocks[stage_num]
        self.use_spatial_attention = params.use_spatial_attention
        self.sa_width = params.enc_sa_width[stage_num]

        self.use_depthwise_attention = params.use_depthwise_attention
        if self.use_depthwise_attention:
            self.key_width = params.query_width
            self.lw_v = layers.LayerNormalization(axis=[1, 2, 3])

        # For the encoder, a block is just a ResNet Block
        out_width = self.in_width
        self.blocks = Sequential()
        for i in range(self.num_blocks):
            out_width = self.in_width
            # If this is the final block of the stage, we also output the keys k^q_l
            if i == self.num_blocks - 1 and self.use_depthwise_attention:
                out_width += self.key_width

            in_width = self.in_width
            # If this is the first block of the second, third, etc stage, input keys
            # if stage_num > 0 and i == 0 and self.use_depthwise_attention:
            # in_width += self.key_width

            self.blocks.add(
                Block(
                    input_size=self.input_size,
                    in_width=in_width,
                    middle_width=self.middle_width,
                    out_width=out_width,
                    kernelsize=params.kernelsizes[str(self.input_size)],
                    activation=params.block_activation,
                    bn=params.block_bn,
                    residual=True,
                    zero_last=False,
                    model_depth=params.model_depth,
                    depthwise=False,
                    use_attention=self.use_spatial_attention,
                    attention_width=self.sa_width,
                    dwa_enc=self.use_depthwise_attention,
                )
            )

        self.pool = PoolLayer(
            input_size=self.input_size,
            in_width=out_width,
            out_width=self.pool_width,
            pool_activation=params.pool_activation,
            unpool=False,
        )

    def build(self):
        for enc_block in self.blocks.layers:
            enc_block.build()
        self.pool.build()

        in_width = self.in_width
        # if self.stage_num > 0 and self.use_depthwise_attention:
        #     in_width += self.key_width

        _, _ = self.call(random.normal([1, self.input_size, self.input_size, in_width]))

    def call(self, x):
        for enc_block in self.blocks.layers:
            # Only the final block will have x with more channels
            x = enc_block(x)

        # return pooled x for the next stage and the activations for the decoder
        if self.use_depthwise_attention:
            if self.num_blocks > 0:
                vq = x[:, :, :, : self.in_width]  # (B, H, W, in_width)
                vq += ops.gelu(self.lw_v(vq))
                kq = x[:, :, :, self.in_width :]  # (B, H, W, key_width)
            else:
                vq = None
                kq = None
            return self.pool(x), [vq, kq]
        else:
            return self.pool(x), x


class DecBlock(layers.Layer):
    """
    Flow layers and attention are not available in this code snippet.
    This means that all flow-related and attention-related arguments are omitted.
    The functions are still there to allow for compatibility with the code origin.
    """

    def __init__(
        self,
        input_size,
        in_width,
        middle_width,
        zdim,
        z_out,
        kernelsize,
        block_activation,
        block_bn,
        model_depth,
        depthwise,
        num_flows,
        flow_in_ch,
        num_ortho_vecs,
        flow_type,
        spectral_norm,
        convsylv_channels,
        convsylv_flows_per_stage,
        convsylv_splitfirst,
        convsylv_stage_limit,
        gradient_smoothing,
        use_spatial_attention,
        sa_width,
        use_depthwise_attention,
        query_width,
        num_queries,
        first_block,
        last_block,
    ):
        super().__init__()
        self.first_block = first_block
        self.last_block = last_block
        # The blocks cannot have an identical seedgenerator, they also cannot share a single one,
        # this is necessary because of stateless mirrorstrategy
        seed = pythonrandom.randint(0, 1_000_000)
        self.seed_gen = random.SeedGenerator(1337 + seed)

        # Parameters used for building
        self.input_size = input_size
        self.in_width = in_width
        self.middle_width = middle_width
        self.zdim = zdim
        self.model_depth = model_depth

        self.use_spatial_attention = use_spatial_attention
        self.sa_width = sa_width
        self.use_depthwise_attention = use_depthwise_attention

        self.num_flows = num_flows
        self.flow_in_ch = flow_in_ch
        self.num_ortho_vecs = num_ortho_vecs
        self.flow_type = flow_type

        if self.flow_type != "none" and self.num_flows > 0:
            self.use_flow = True
        else:
            self.use_flow = False

        p_out_width = 2 * self.zdim + in_width
        if self.use_depthwise_attention:
            self.query_width = query_width
            self.num_queries = num_queries
            p_out_width += self.query_width

            self.combine_queries = self.num_queries > 1
            if self.combine_queries:
                self.queries_comb_q = layers.Conv2D(self.in_width, kernel_size=1)
                if not self.first_block:
                    self.queries_comb_p = layers.Conv2D(self.in_width, kernel_size=1)

            queries_out = self.query_width * num_queries
            self.gamma_q = Variable(0, trainable=True, dtype="float32")

            if not self.first_block:
                # Also output prior queries
                queries_out *= 2
                self.gamma_p = Variable(0, trainable=True, dtype="float32")

            self.queries = Block(
                input_size=self.input_size,
                in_width=self.in_width,
                middle_width=self.middle_width,
                out_width=queries_out,
                kernelsize=kernelsize,
                activation=block_activation,
                bn=block_bn,
                residual=False,
                zero_last=False,
                model_depth=model_depth,
                depthwise=depthwise,
                use_attention=use_spatial_attention,
                attention_width=sa_width,
            )

        q_out_width = self.zdim * 2
        if self.flow_type == "sylvester":
            q_out_width += self.flow_in_ch
        # Block that takes activations from encoder
        self.q = Block(
            input_size=self.input_size,
            in_width=2 * self.in_width,
            middle_width=self.middle_width,
            out_width=q_out_width,
            kernelsize=kernelsize,
            activation=block_activation,
            bn=block_bn,
            residual=False,
            zero_last=False,
            model_depth=model_depth,
            depthwise=depthwise,
            use_attention=use_spatial_attention,
            attention_width=sa_width,
        )
        if self.use_flow:
            if self.flow_type == "sylvester":
                self.flows = OrthogonalSylvesterStack(
                    num_flows=self.num_flows,
                    num_ortho_vecs=self.num_ortho_vecs,
                    z_ch=self.zdim,
                    build_shape=[1, self.input_size, self.input_size, self.flow_in_ch],
                    model_depth=model_depth,
                )
            elif self.flow_type == "conv_sylvester":
                self.flows = ConvSylvester(
                    input_size=(self.input_size, self.input_size, self.zdim),
                    flows_per_level=convsylv_flows_per_stage,
                    spectral_norm=spectral_norm,
                    sylv_channels=convsylv_channels,
                    split_first=convsylv_splitfirst,
                    stage_limit=convsylv_stage_limit,
                )
            self.kl = flow_kl()
        else:
            self.kl = GaussianAnalyticalKL()

        self.p = Block(
            input_size=self.input_size,
            in_width=self.in_width,
            middle_width=self.middle_width,
            out_width=p_out_width,
            kernelsize=kernelsize,
            activation=block_activation,
            bn=block_bn,
            residual=False,
            zero_last=True,
            model_depth=model_depth,
            depthwise=depthwise,
            use_attention=use_spatial_attention,
            attention_width=sa_width,
        )

        # Residual block for after adding Z
        if not self.last_block:
            self.res = Block(
                input_size=self.input_size,
                in_width=self.in_width,
                middle_width=self.middle_width,
                out_width=self.in_width,
                kernelsize=kernelsize,
                activation=block_activation,
                bn=block_bn,
                residual=True,
                zero_last=False,
                model_depth=model_depth,
                depthwise=depthwise,
                use_attention=use_spatial_attention,
                attention_width=sa_width,
            )
            self.z_proj = layers.Conv2D(in_width, kernel_size=1)

        self.sp = SoftPlus(gradient_smoothing)
        self.z_out_f = layers.Conv2D(z_out[-1], kernel_size=1)
        if z_out[0] > input_size:
            self.z_out_up = layers.UpSampling2D(
                size=z_out[0] // input_size, interpolation="nearest"
            )
        else:
            self.z_out_up = layers.Identity()

    def build(self):
        self.q.build()
        self.p.build()
        if self.use_flow:
            self.flows.build()

        if not self.last_block:
            self.res.build()
            _ = self.z_proj(random.normal([1, self.input_size, self.input_size, self.zdim]))
            weights = self.z_proj.get_weights()
            weights[0] *= np.sqrt(1 / self.model_depth)
            self.z_proj.set_weights(weights)

    def sample(self, x, act):
        # Calculate all attention outputs
        if self.use_depthwise_attention:
            x, vp, kp = x
            vq, kq = act

            queries = self.queries(x)

            if not self.first_block:
                qq, qp = ops.split(queries, 2, axis=3)
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qp, self.num_queries, axis=-1)):
                        attn = attention_block(query, kp, vp)
                        if i == 0:
                            attn_p = attn
                        else:
                            attn_p = ops.concatenate([attn_p, attn], axis=-1)
                    attn_p = self.queries_comb_p(attn_p)
                else:
                    attn_p = attention_block(qp, kp, vp)
                attn_p += ops.gelu(attn_p)
            else:
                qq, qp = queries, None

            if ops.shape(vq)[-1] > 1:
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qq, self.num_queries, axis=-1)):
                        attn = attention_block(query, kq[:, :, :, :, :-1], vq[:, :, :, :, :-1])
                        if i == 0:
                            attn_q = attn
                        else:
                            attn_q = ops.concatenate([attn_q, attn], axis=-1)
                    attn_q = self.queries_comb_q(attn_q)
                else:
                    attn_q = attention_block(qq, kq[:, :, :, :, :-1], vq[:, :, :, :, :-1])
                attn_q += ops.gelu(attn_q)
            else:
                attn_q = ops.zeros_like(x)
            q_out = self.q(
                ops.concatenate([x, vq[:, :, :, :, -1] + self.gamma_q * attn_q], axis=-1)
            )
        else:
            q_out = self.q(ops.concatenate([x, act], axis=-1))

        if self.flow_type == "sylvester":
            qm, q_std, h = (
                q_out[:, :, :, : self.zdim],
                q_out[:, :, :, self.zdim : self.zdim * 2],
                q_out[:, :, :, self.zdim * 2 :],
            )
        else:
            qm, q_std = ops.split(q_out, 2, axis=3)
            h = None

        # calculate prior (mu_p, variance_p, and residual)
        if self.use_depthwise_attention:
            if self.first_block:
                prior = self.p(x)
            else:
                prior = self.p(x + self.gamma_p * attn_p)
            pm, p_std, vpl, kpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 : self.zdim * 2 + self.in_width],
                prior[:, :, :, self.zdim * 2 + self.in_width :],
            )

            x = ops.add(x, vpl)  # (B, H, W, in_width)

            if self.first_block:
                vp = ops.expand_dims(vpl, axis=-1)
                kp = ops.expand_dims(kpl, axis=-1)
            else:
                vp = ops.concatenate([vp, ops.expand_dims(vpl, axis=-1)], axis=-1)
                kp = ops.concatenate([kp, ops.expand_dims(kpl, axis=-1)], axis=-1)
            x = (x, vp, kp)

        else:
            prior = self.p(x)
            pm, p_std, vpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 :],
            )
            x = ops.add(x, vpl)

        q_std = self.sp(q_std)
        p_std = self.sp(p_std)

        noise = random.normal(ops.shape(q_std), seed=self.seed_gen)

        z0 = ops.add(qm, ops.multiply(q_std, noise))

        if self.use_flow:
            zk, log_det_j = self.flows.call(z0, h)
            kl = self.kl.call(qm, q_std, pm, p_std, z0, zk)
            return x, zk, (kl, log_det_j)
        else:
            kl = self.kl.call(qm, q_std, pm, p_std)
            return x, z0, kl

    def sample_uncond(self, x, t=1):
        if self.use_depthwise_attention:
            x, vp, kp = x
            if not self.first_block:
                queries = self.queries(x)
                _, qp = ops.split(queries, 2, axis=3)
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qp, self.num_queries, axis=-1)):
                        attn = attention_block(query, kp, vp)
                        if i == 0:
                            attn_p = attn
                        else:
                            attn_p = ops.concatenate([attn_p, attn], axis=-1)
                    attn_p = self.queries_comb_p(attn_p)
                else:
                    attn_p = attention_block(qp, kp, vp)
                attn_p += ops.gelu(attn_p)

                prior = self.p(x + self.gamma_p * attn_p)
            else:
                prior = self.p(x)

            pm, p_std, vpl, kpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 : self.zdim * 2 + self.in_width],
                prior[:, :, :, self.zdim * 2 + self.in_width :],
            )

            x = ops.add(x, vpl)  # (B, H, W, in_width)

            if self.first_block:
                vp = ops.expand_dims(vpl, axis=-1)
                kp = ops.expand_dims(kpl, axis=-1)
            else:
                vp = ops.concatenate([vp, ops.expand_dims(vpl, axis=-1)], axis=-1)
                kp = ops.concatenate([kp, ops.expand_dims(kpl, axis=-1)], axis=-1)
            x = (x, vp, kp)
        else:
            prior = self.p(x)
            pm, p_std, vpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 :],
            )
            x = ops.add(x, vpl)

        p_std = self.sp(p_std)
        noise = random.normal(ops.shape(p_std), seed=self.seed_gen)
        z = pm + p_std * noise * t
        return x, z

    def call(self, x, act):
        x, z, kl = self.sample(x, act)
        if self.use_depthwise_attention:
            x, vp, kp = x
        if not self.last_block:
            x = ops.add(x, self.z_proj(z))
            x = self.res(x)
        if self.use_depthwise_attention:
            x = (x, vp, kp)

        z = self.z_out_f(z)
        z = self.z_out_up(z)
        return x, z, kl

    def call_uncond(self, x, t=1):
        x, z = self.sample_uncond(x, t)
        if self.use_depthwise_attention:
            x, vp, kp = x
        if not self.last_block:
            x = ops.add(x, self.z_proj(z))
            x = self.res(x)
        if self.use_depthwise_attention:
            x = (x, vp, kp)

        z = self.z_out_f(z)
        z = self.z_out_up(z)
        return x, z


class Block(layers.Layer):
    """
    This class represents a single ResNet block and all the options it can have.
    Again, self-attention is not implemented in this snippet.
    """

    def __init__(
        self,
        input_size,
        in_width,
        middle_width,
        out_width,
        kernelsize,
        activation,
        bn,
        residual,
        zero_last,
        model_depth,
        depthwise,
        use_attention,
        attention_width,
        dwa_enc=False,
    ):
        super().__init__()
        # Parameters used for building
        self.input_size = input_size
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.zero_last = zero_last
        self.model_depth = model_depth
        self.depthwise = depthwise

        # Parameters used for calling
        self.activation = activation
        self.residual = residual
        self.use_attention = use_attention and (attention_width > 0)
        if self.use_attention:
            self.attention_width = attention_width
        self.dwa_enc = dwa_enc

        # Groupnorm layers
        self.gn1 = layers.GroupNormalization(groups=in_width // 8) if bn else layers.Identity()
        self.gn2 = layers.GroupNormalization(groups=middle_width // 8) if bn else layers.Identity()
        self.gn3 = layers.GroupNormalization(groups=middle_width // 8) if bn else layers.Identity()
        self.gn4 = layers.GroupNormalization(groups=middle_width // 8) if bn else layers.Identity()

        # Convolutional layers
        self.c1 = layers.Conv2D(middle_width, kernel_size=1)
        if depthwise:
            # Depthwise 5x5 convs as used in NVAE
            self.c2 = (
                layers.DepthwiseConv2D(kernel_size=5, padding="same")
                if (self.input_size > 4)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
            self.c3 = (
                layers.DepthwiseConv2D(kernel_size=5, padding="same")
                if (self.input_size > 4)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
        else:
            # 3x3 convs as used in (Efficient)-VDVAE
            self.c2 = layers.Conv2D(middle_width, kernel_size=kernelsize, padding="same")
            self.c3 = layers.Conv2D(middle_width, kernel_size=kernelsize, padding="same")
        kern_init = "zeros" if zero_last else "glorot_uniform"
        self.c4 = layers.Conv2D(out_width, kernel_size=1, kernel_initializer=kern_init)

        # SPATIAL-SELF-ATTENTION for deep attentive variational inference
        if self.use_attention:
            pass

    def build(self):
        _ = self.call(random.normal([1, self.input_size, self.input_size, self.in_width]))

        # Initialize last layer with 1/sqrt(n)
        weights = self.c4.get_weights()
        weights[0] *= np.sqrt(1 / self.model_depth)
        self.c4.set_weights(weights)

    def call(self, x):
        xhat = self.c1(self.activation(self.gn1(x)))
        if self.use_attention:
            xhat = self.attention1(xhat)

        xhat = self.c2(self.activation(self.gn2(xhat)))
        if self.use_attention:
            xhat = self.attention2(xhat)

        xhat = self.c3(self.activation(self.gn3(xhat)))
        xhat = self.c4(self.activation(self.gn4(xhat)))

        if self.dwa_enc and self.in_width < self.out_width:
            out = ops.concatenate(
                [
                    ops.add(xhat[:, :, :, : self.in_width], x),  # vq
                    xhat[:, :, :, self.in_width :],  # kq
                ],
                axis=-1,
            )
        elif self.dwa_enc and self.in_width > self.out_width:
            out = ops.add(xhat, x[:, :, :, : self.out_width])  # vq

        else:
            out = ops.add(x, xhat) if self.residual else xhat
        return out


class PoolLayer(layers.Layer):
    """
    Pooling layer that can perform either:
    AveragePooling2D for the encoder,
    Upsampling2D for the decoder.
    """

    def __init__(self, input_size, in_width, out_width, pool_activation, unpool, data_size=None):
        super().__init__()
        # Building
        self.input_size = input_size
        self.in_width = in_width
        self.out_width = out_width
        self.activation = pool_activation

        # Layers
        if in_width == out_width:
            self.c1 = layers.Identity()
        else:
            self.c1 = layers.Conv2D(out_width, kernel_size=1)

        if unpool:
            self.pool = layers.UpSampling2D(2) if (input_size < data_size) else layers.Identity()
        else:
            self.pool = layers.AveragePooling2D(2) if (input_size > 1) else layers.Identity()

    def build(self):
        _ = self.call(random.normal([1, self.input_size, self.input_size, self.in_width]))

    def call(self, x):
        return self.pool(self.activation(self.c1(x)))
