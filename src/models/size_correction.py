"""
InsertAnything + Depth ControlNet (Size Correction)

Extends InsertAnything by adding a frozen Flux ControlNet for depth conditioning.
The depth map provides structural guidance for more precise object insertion
with correct size and spatial placement.

Architecture:
- FluxFillPipeline: base inpainting model (with LoRA training)
- FluxPriorReduxPipeline: reference image encoder (frozen)
- FluxControlNetModel: depth conditioning (frozen)

During training, only the LoRA layers on the FluxFill transformer are updated.
The ControlNet produces residual block samples that are injected into the
transformer's forward pass (already supported by the existing transformer code).
"""

import os
import lightning as L
from diffusers.pipelines import FluxFillPipeline, FluxPriorReduxPipeline
from diffusers.models import FluxControlNetModel
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict

from PIL import Image
from .transformer import tranformer_forward
from .pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
from .image_project import image_output


def mask_controlnet_residuals_for_diptych(
    controlnet_block_samples,
    controlnet_single_block_samples,
    img_ids,
    dtype,
):
    """
    Zero out ControlNet residuals for the left half (reference side) of the diptych.

    The depth diptych has all-black padding on the left half. ControlNet residuals
    from this region carry a gray bias that contaminates the generated image.
    Masking them to zero confines depth guidance to the target (right) half only.

    Args:
        controlnet_block_samples: list of [B, N, D] from ControlNet double blocks
        controlnet_single_block_samples: list of [B, N, D] from ControlNet single blocks
        img_ids: [N, 3] packed position IDs — columns are (batch_id, y, x)
        dtype: target dtype for the mask

    Returns:
        masked (controlnet_block_samples, controlnet_single_block_samples)
    """
    x_coords = img_ids[:, 2]
    grid_w = int(x_coords.max().item()) + 1
    right_half = (x_coords >= grid_w // 2).to(dtype)
    spatial_mask = right_half.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

    controlnet_block_samples = [
        block * spatial_mask for block in controlnet_block_samples
    ]
    controlnet_single_block_samples = [
        block * spatial_mask for block in controlnet_single_block_samples
    ]
    return controlnet_block_samples, controlnet_single_block_samples


def encode_depth_for_controlnet(pipeline, depth_images):
    """
    Encode depth images into latent space for ControlNet conditioning.
    
    The depth image needs to be VAE-encoded and packed into the same latent format
    as the main hidden_states so the ControlNet can process it.
    
    Args:
        pipeline: FluxFillPipeline (provides VAE, image_processor, etc.)
        depth_images: tensor of depth images [B, 3, H, W] in [0, 1] range
    
    Returns:
        depth_latents: packed latent representation
        depth_ids: positional IDs for the depth latents
    """
    # Preprocess depth images through the VAE
    depth_images = pipeline.image_processor.preprocess(depth_images)
    depth_images = depth_images.to(pipeline.device).to(pipeline.dtype)
    depth_latents = pipeline.vae.encode(depth_images).latent_dist.sample()
    depth_latents = (
        depth_latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    
    # Pack into flux latent format
    depth_latents_packed = pipeline._pack_latents(depth_latents, *depth_latents.shape)
    depth_ids = pipeline._prepare_latent_image_ids(
        depth_latents.shape[0],
        depth_latents.shape[2],
        depth_latents.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    
    if depth_latents_packed.shape[1] != depth_ids.shape[0]:
        depth_ids = pipeline._prepare_latent_image_ids(
            depth_latents.shape[0],
            depth_latents.shape[2] // 2,
            depth_latents.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    
    return depth_latents_packed, depth_ids


# Packed VAE tokens last dim (Flux Fill)
HF_LATENT_DIM = 64
HF_INJECT_CKPT_NAME = "hf_latent_inject.pt"


def decode_packed_latents_to_rgb01(pipeline, packed_tokens: torch.Tensor, height_px: int, width_px: int):
    """packed_tokens: [B, N, C] VAE-packed latents -> image [B, 3, H, W] in [0, 1]."""
    latents = pipeline._unpack_latents(
        packed_tokens,
        height_px,
        width_px,
        pipeline.vae_scale_factor,
    )
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(pipeline.vae.dtype)
    img = pipeline.vae.decode(latents, return_dict=False)[0]
    img = (img.float() + 1.0) / 2.0
    return img.clamp(0, 1)


def high_frequency_map_torch(x: torch.Tensor, radius_frac: float = 0.15) -> torch.Tensor:
    """x: B,3,H,W in [0,1]. Differentiable HiFi-style high-pass (per channel), max-norm per image.

    Internally promotes to float32 for FFT numerical stability (bfloat16 FFT
    can produce NaN/Inf), then casts back to the input dtype.
    """
    orig_dtype = x.dtype
    x = x.float()
    B, C, H, W = x.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = max(1, int(radius_frac * min(H, W) / 2))
    hp_mask = (dist > float(r)).float()
    out = torch.zeros_like(x)
    for c in range(C):
        xc = x[:, c]
        F = torch.fft.fft2(xc, dim=(-2, -1))
        Fshift = torch.fft.fftshift(F, dim=(-2, -1))
        Fh = Fshift * hp_mask
        F2 = torch.fft.ifftshift(Fh, dim=(-2, -1))
        out[:, c] = torch.abs(torch.fft.ifft2(F2, dim=(-2, -1)))
    out = out / (out.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return out.to(orig_dtype)


class InsertAnythingDepth(L.LightningModule):
    """
    InsertAnything with Depth ControlNet for size-corrected object insertion.
    
    Compared to the original InsertAnything:
    - Adds a frozen FluxControlNetModel for depth conditioning
    - Training data includes depth maps of the target scene
    - The depth ControlNet provides structural/spatial guidance
    - Only LoRA layers on the main transformer are trained
    """
    
    def __init__(
        self,
        flux_fill_id: str,
        flux_redux_id: str,
        controlnet_depth_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        controlnet_conditioning_scale: float = 0.5,
        pretrained_lora_path: str = None,
        train_controlnet: bool = True,
        use_hf_inject: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.train_controlnet = train_controlnet
        self.use_hf_inject = use_hf_inject

        # ============ Load FluxFill Pipeline ============
        self.flux_fill_pipe: FluxFillPipeline = (
            FluxFillPipeline.from_pretrained(flux_fill_id, low_cpu_mem_usage=True).to(dtype=dtype).to(device)
        )
        # print("🗑️ Freeing Text Encoders to save memory...")
        # del self.flux_fill_pipe.text_encoder
        # del self.flux_fill_pipe.text_encoder_2
        # del self.flux_fill_pipe.tokenizer
        # del self.flux_fill_pipe.tokenizer_2

        # ============ Load pretrained LoRA & merge into base weights ============
        # This allows us to continue training from a pre-trained InsertAnything LoRA
        # by merging it into the base model first, then adding a fresh LoRA on top.
        if pretrained_lora_path:
            print(f"📦 Loading pretrained LoRA: {pretrained_lora_path}", flush=True)
            self.flux_fill_pipe.load_lora_weights(pretrained_lora_path)
            self.flux_fill_pipe.fuse_lora()
            self.flux_fill_pipe.unload_lora_weights()
            print("✅ Pretrained LoRA merged into base weights.", flush=True)

        # ============ Load Redux (reference image encoder) ============
        self.flux_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(flux_redux_id).to(dtype=dtype).to(device)
        )
        self.flux_redux.image_embedder.requires_grad_(False).eval()
        self.flux_redux.image_encoder.requires_grad_(False).eval()

        # ============ Load Depth ControlNet ============
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_depth_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        if self.train_controlnet:
            # Unfreeze ControlNet — let it adapt to FluxFill's feature space
            # The pre-trained Shakker-Labs ControlNet was trained on vanilla FLUX.1-dev,
            # but FluxFill has diverged (inpainting fine-tune). The residuals don't match
            # FluxFill's internal representations, so we need to fine-tune the ControlNet.
            self.controlnet.requires_grad_(True)
            self.controlnet.train()
            if gradient_checkpointing:
                self.controlnet.gradient_checkpointing = True
            print("🔓 ControlNet UNFROZEN — will be trained jointly.", flush=True)
        else:
            self.controlnet.requires_grad_(False).eval()
            print("🔒 ControlNet FROZEN.", flush=True)

        # ============ Setup Transformer ============
        self.transformer = self.flux_fill_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze encoders and VAE
        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        # Optional: HF latent -> inner-dim residual (does NOT change x_embedder shape; old LoRA ckpts load as-is)
        _odim = int(self.transformer.x_embedder.out_features)
        if use_hf_inject:
            self.hf_latent_inject = nn.Linear(HF_LATENT_DIM, _odim, bias=False).to(
                device=device, dtype=dtype
            )
            nn.init.zeros_(self.hf_latent_inject.weight)
        else:
            self.hf_latent_inject = None

        # Initialize LoRA layers (fresh adapter on top of merged weights)
        self.lora_layers = self.init_lora(lora_path, lora_config)

        if self.hf_latent_inject is not None and lora_path:
            _dir = lora_path if os.path.isdir(lora_path) else os.path.dirname(lora_path)
            if _dir:
                _hp = os.path.join(_dir, HF_INJECT_CKPT_NAME)
                if os.path.isfile(_hp):
                    self.hf_latent_inject.load_state_dict(torch.load(_hp, map_location=device))
                    print(f"✅ Loaded {HF_INJECT_CKPT_NAME} (HF inject)", flush=True)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        """Initialize LoRA adapter on the transformer.
        
        Args:
            lora_path: Path to a saved LoRA checkpoint (from our own training).
                       If provided, will add a LoRA adapter with `lora_config` and
                       then load the saved weights into it.
            lora_config: Config dict for LoraConfig (rank, alpha, target_modules, etc.).
                         Always required — defines the adapter architecture.
        """
        assert lora_config, "lora_config is required to define the LoRA adapter"
        # Always add the adapter first (defines architecture)
        self.transformer.add_adapter(LoraConfig(**lora_config))

        if lora_path:
            # Load weights from a previous checkpoint of OUR training
            print(f"🔄 Loading LoRA checkpoint: {lora_path}", flush=True)
            from safetensors.torch import load_file
            import os
            lora_file = os.path.join(lora_path, "pytorch_lora_weights.safetensors") \
                if os.path.isdir(lora_path) else lora_path
            state_dict = load_file(lora_file)

            # save_lora uses get_peft_model_state_dict() which outputs diffusers-format keys
            # e.g. "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight"
            # But set_peft_model_state_dict expects peft-format keys
            # e.g. "base_model.model.single_transformer_blocks.0.attn.to_q.lora_A.weight"
            # So we convert: strip "transformer." prefix and add "base_model.model." prefix
            converted = {}
            for k, v in state_dict.items():
                if k.startswith("transformer."):
                    new_key = "base_model.model." + k[len("transformer."):]
                else:
                    new_key = "base_model.model." + k
                converted[new_key] = v

            from peft import set_peft_model_state_dict
            incompatible = set_peft_model_state_dict(self.transformer, converted)
            if incompatible.missing_keys:
                print(f"⚠️  Missing keys: {len(incompatible.missing_keys)}")
                print(f"    Examples: {incompatible.missing_keys[:3]}...")
            if incompatible.unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(incompatible.unexpected_keys)}")
                print(f"    Examples: {incompatible.unexpected_keys[:3]}...")
            if not incompatible.missing_keys and not incompatible.unexpected_keys:
                print("✅ All LoRA keys matched perfectly.", flush=True)
            else:
                print("✅ LoRA weights loaded (with warnings above).", flush=True)

        lora_layers = filter(
            lambda p: p.requires_grad, self.transformer.parameters()
        )
        return list(lora_layers)

    def save_lora(self, path: str, training_state: dict = None):
        """Save LoRA weights, ControlNet weights, and optionally full training state.
        
        Args:
            path: Directory to save checkpoint to.
            training_state: Optional dict containing training state for resumption:
                - global_step: Lightning trainer's global step
                - callback_total_steps: TrainingCallback's total_steps counter
                - log_loss: EMA loss value
                - optimizer_state_dict: Optimizer state dict (for Prodigy, this is critical)
        """
        os.makedirs(path, exist_ok=True)
        
        # 1. Save LoRA weights
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        
        # 2. Save ControlNet weights if it's being trained
        if self.train_controlnet:
            controlnet_path = os.path.join(path, "controlnet")
            self.controlnet.save_pretrained(controlnet_path)
            print(f"💾 ControlNet saved to {controlnet_path}", flush=True)
        
        # 3. Save training state for full resumption
        if training_state is not None:
            state_path = os.path.join(path, "training_state.pt")
            torch.save(training_state, state_path)
            print(f"💾 Training state saved to {state_path} "
                  f"(step={training_state.get('global_step', '?')})", flush=True)

        if self.hf_latent_inject is not None:
            torch.save(self.hf_latent_inject.state_dict(), os.path.join(path, HF_INJECT_CKPT_NAME))
            print(f"💾 HF inject saved to {os.path.join(path, HF_INJECT_CKPT_NAME)}", flush=True)

    def configure_optimizers(self):
        # Freeze the transformer (LoRA params will be unfrozen below)
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = list(self.lora_layers)

        # Unfreeze trainable LoRA parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        if self.hf_latent_inject is not None:
            for p in self.hf_latent_inject.parameters():
                p.requires_grad_(True)
            self.trainable_params = self.trainable_params + list(self.hf_latent_inject.parameters())

        # Include ControlNet parameters if training it
        _hf = " + HF inject" if self.hf_latent_inject is not None else ""
        if self.train_controlnet:
            controlnet_params = [p for p in self.controlnet.parameters() if p.requires_grad]
            self.trainable_params = self.trainable_params + controlnet_params
            print(
                f"🔧 Trainable params: {len(self.lora_layers)} LoRA{_hf} + {len(controlnet_params)} ControlNet",
                flush=True,
            )
        else:
            print(
                f"🔧 Trainable params: {len(self.lora_layers)} LoRA{_hf} (ControlNet frozen)",
                flush=True,
            )

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError(f"Optimizer {opt_config['type']} not supported")

        return optimizer

    def training_step(self, batch, batch_idx):
        tc = getattr(self.trainer, "training_config", {})
        depth_dropout_prob = tc.get("depth_dropout_prob", 0.0)
        dal_lambda = float(tc.get("dal_lambda", 0.0))
        hf_hp_radius = float(tc.get("hf_hp_radius", 0.15))
        dal_t_max = float(tc.get("dal_t_max", 0.5))
        step_loss = self.step(
            batch,
            depth_dropout_prob=depth_dropout_prob,
            dal_lambda=dal_lambda,
            hf_hp_radius=hf_hp_radius,
            dal_t_max=dal_t_max,
        )
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(
        self,
        batch,
        depth_dropout_prob: float = 0.0,
        dal_lambda: float = 0.0,
        hf_hp_radius: float = 0.15,
        dal_t_max: float = 0.5,
    ):
        """
        Training step with depth conditioning.
        
        Args:
            batch: dict with keys "result", "src", "mask", "ref", "depth"
            depth_dropout_prob: Probability of zeroing out depth ControlNet residuals
                per sample. Only used during training. Default 0.0 (no dropout)
                means depth is always active — safe for inference.
            dal_lambda: Weight for pixel-space DAL (0 = off). Decodes predicted x0
                through VAE and compares high-frequency maps in pixel space, following
                HiFi-Inpaint (CVPR 2026).
            hf_hp_radius: High-pass radius for HF extraction (~0.1–0.2).
            dal_t_max: Only apply DAL when t < this value (x0 estimate is unreliable
                at high noise levels). 0.5 is a good default.
        
        batch contains:
            - "result": target diptych (ref | target) [B, 3, H, 2W]
            - "src": source diptych (ref | masked_target) [B, 3, H, 2W]
            - "mask": mask diptych [B, 3, H, 2W]
            - "ref": reference image [B, 3, H, W]
            - "depth": depth map diptych (black | depth) [B, 3, H, 2W]
            - "hf_diptych" (optional): ref HF | zeros, for HF inject [B, 3, H, 2W]
        """
        imgs = batch["result"]
        src = batch["src"]
        mask = batch["mask"]
        ref = batch["ref"]
        depth = batch["depth"]

        # ============ Encode reference image via Redux ============
        prompt_embeds = []
        pooled_prompt_embeds = []

        for i in range(ref.shape[0]):
            image_tensor = ref[i].cpu()
            image_tensor = image_tensor.permute(1, 2, 0)
            image_numpy = image_tensor.numpy()
            pil_image = Image.fromarray((image_numpy * 255).astype('uint8'))

            prompt_embed, pooled_prompt_embed = image_output(
                self.flux_redux, pil_image, self.device
            )
            prompt_embeds.append(prompt_embed.squeeze(1))
            pooled_prompt_embeds.append(pooled_prompt_embed.squeeze(1))

        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)

        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe,
            prompt_embeds=prompt_embeds.to(self.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(self.device),
        )

        # ============ Prepare main training inputs ============
        hf_latents = None
        with torch.no_grad():
            # Encode target image
            x_0, img_ids = encode_images(self.flux_fill_pipe, imgs)

            # Prepare noise schedule
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Encode masked source + mask
            src_latents, mask_latents = Flux_fill_encode_masks_images(
                self.flux_fill_pipe, src, mask
            )
            condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

            # Guidance embedding
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

            # Encode depth diptych into latents for controlnet_cond
            depth_latents, depth_ids = encode_depth_for_controlnet(
                self.flux_fill_pipe, depth
            )

            # HF diptych (VAE + pack, same path as depth) — optional conditioning
            if self.hf_latent_inject is not None and batch.get("hf_diptych") is not None:
                hf_latents, _ = encode_depth_for_controlnet(
                    self.flux_fill_pipe, batch["hf_diptych"]
                )

        image_cond_residual = None
        if self.hf_latent_inject is not None and hf_latents is not None:
            image_cond_residual = self.hf_latent_inject(hf_latents)

        # ============ Depth ControlNet forward ============
        # When train_controlnet=True, gradients flow through ControlNet
        # When train_controlnet=False, this is wrapped in torch.no_grad() via eval mode
        #
        # ControlNet input/output:
        #   hidden_states  = x_t only (64-dim packed noisy latent)
        #   controlnet_cond = depth_latents (64-dim packed depth latent)
        #   output: residuals in 3072-dim (inner_dim), matching transformer hidden states
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=x_t,
            controlnet_cond=depth_latents,
            conditioning_scale=self.controlnet_conditioning_scale,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=t.to(x_t.dtype),
            img_ids=img_ids if img_ids.ndim == 2 else img_ids[0],
            txt_ids=text_ids if text_ids.ndim == 2 else text_ids[0],
            guidance=guidance.to(x_t.dtype) if guidance is not None else None,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        # ============ Depth Dropout ============
        # With some probability, zero out ControlNet residuals so the model
        # learns to inpaint correctly even without depth guidance.
        # This is per-sample: each sample in the batch independently drops.
        # NOTE: depth_dropout_prob is passed in from training_step(), NOT stored
        # on the model. This ensures inference (callback visualization, etc.)
        # never accidentally drops depth — it always defaults to 0.0.
        if depth_dropout_prob > 0:
            drop_mask = torch.rand(imgs.shape[0], device=self.device) < depth_dropout_prob
            if drop_mask.any():
                # Zero out controlnet residuals for dropped samples
                controlnet_block_samples = [
                    block * (~drop_mask).to(block.dtype).view(-1, *([1] * (block.ndim - 1)))
                    for block in controlnet_block_samples
                ]
                controlnet_single_block_samples = [
                    block * (~drop_mask).to(block.dtype).view(-1, *([1] * (block.ndim - 1)))
                    for block in controlnet_single_block_samples
                ]

        # ============ Spatial masking: zero out left-half (reference) residuals ============
        controlnet_block_samples, controlnet_single_block_samples = \
            mask_controlnet_residuals_for_diptych(
                controlnet_block_samples, controlnet_single_block_samples,
                img_ids if img_ids.ndim == 2 else img_ids[0],
                x_t.dtype,
            )

        # ============ Transformer forward with ControlNet residuals ============
        transformer_out = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            hidden_states=torch.cat((x_t, condition_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            image_cond_residual=image_cond_residual,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute flow-matching loss
        target = (x_1 - x_0)
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")

        # ============ Pixel-Space Detail-Aware Loss (HiFi-Inpaint, CVPR 2026) ============
        # L_DA = || H(I_pred) * M  -  H(I_gt) * M ||^2
        #
        # Gradient path: pred -> x0_hat -> VAE.decode -> HF_map -> L_DA
        # VAE decoder is frozen but differentiable; FFT ops are differentiable.
        # Only applied when t < dal_t_max because the x0 estimate from flow
        # matching is unreliable at high noise levels.
        #
        # Memory optimizations:
        #   1. Downscale latents before VAE decode (dal_downsample > 1) to reduce
        #      pixel-space activation memory from O(H*W) to O(H/s * W/s).
        #   2. Use float32 autocast only for FFT; VAE decode stays in model dtype.
        #   3. Detach intermediate pixel images after HF extraction to free the
        #      full-resolution decode graph early (grad still flows via chain rule
        #      on the downscaled path).
        if dal_lambda > 0.0:
            dal_mask = (t < dal_t_max)
            if dal_mask.any():
                t_sel = t_[dal_mask]
                x_0_hat = x_t[dal_mask] - t_sel * pred[dal_mask]

                height_px, width_px = imgs.shape[2], imgs.shape[3]
                ds = int(getattr(self.trainer, "training_config", {}).get("dal_downsample", 2))

                if ds > 1:
                    dal_h, dal_w = height_px // ds, width_px // ds
                    vae_dtype = self.flux_fill_pipe.vae.dtype
                    latents_pred = self.flux_fill_pipe._unpack_latents(
                        x_0_hat, height_px, width_px,
                        self.flux_fill_pipe.vae_scale_factor,
                    )
                    latents_pred = (
                        latents_pred / self.flux_fill_pipe.vae.config.scaling_factor
                    ) + self.flux_fill_pipe.vae.config.shift_factor
                    latents_pred = torch.nn.functional.interpolate(
                        latents_pred.float(), scale_factor=1.0 / ds,
                        mode="bilinear", align_corners=False,
                    ).to(vae_dtype)
                    img_pred = self.flux_fill_pipe.vae.decode(
                        latents_pred, return_dict=False
                    )[0]
                    img_pred = ((img_pred.float() + 1.0) / 2.0).clamp(0, 1)

                    with torch.no_grad():
                        latents_gt = self.flux_fill_pipe._unpack_latents(
                            x_0[dal_mask], height_px, width_px,
                            self.flux_fill_pipe.vae_scale_factor,
                        )
                        latents_gt = (
                            latents_gt / self.flux_fill_pipe.vae.config.scaling_factor
                        ) + self.flux_fill_pipe.vae.config.shift_factor
                        latents_gt = torch.nn.functional.interpolate(
                            latents_gt.float(), scale_factor=1.0 / ds,
                            mode="bilinear", align_corners=False,
                        ).to(vae_dtype)
                        img_gt = self.flux_fill_pipe.vae.decode(
                            latents_gt, return_dict=False
                        )[0]
                        img_gt = ((img_gt.float() + 1.0) / 2.0).clamp(0, 1)
                else:
                    dal_h, dal_w = height_px, width_px
                    img_pred = decode_packed_latents_to_rgb01(
                        self.flux_fill_pipe, x_0_hat, height_px, width_px
                    )
                    with torch.no_grad():
                        img_gt = decode_packed_latents_to_rgb01(
                            self.flux_fill_pipe, x_0[dal_mask], height_px, width_px
                        )

                with torch.no_grad():
                    mask_diptych = batch["mask"][dal_mask]
                    pixel_mask = (mask_diptych[:, 0:1] > 0.5).float()
                    pixel_mask = torch.nn.functional.interpolate(
                        pixel_mask, size=(dal_h, dal_w), mode="nearest"
                    )
                    hf_gt_masked = high_frequency_map_torch(
                        img_gt, radius_frac=hf_hp_radius
                    ) * pixel_mask

                hf_pred_masked = high_frequency_map_torch(
                    img_pred, radius_frac=hf_hp_radius
                ) * pixel_mask

                n_mask_pixels = pixel_mask.sum().clamp(min=1.0)
                loss_dal = ((hf_pred_masked - hf_gt_masked) ** 2).sum() / n_mask_pixels
                loss = loss + dal_lambda * loss_dal

        self.last_t = t.mean().item()
        return loss
