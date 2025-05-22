import os
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .utils import get_generator
from .attention_processor import AttnProcessor  # Assuming AttnProcessor is imported as is_torch2_available() is not specified

class BodyProjModel(torch.nn.Module):
    """Projection model for body vector c_body"""
    def __init__(self, cross_attention_dim=768, body_dim=756, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(body_dim, num_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, body_vector):
        # Input: (batch_size, body_dim)
        proj = self.proj(body_vector)  # (batch_size, num_tokens * cross_attention_dim)
        proj = proj.reshape(-1, self.num_tokens, self.cross_attention_dim)  # (batch_size, num_tokens, cross_attention_dim)
        return self.norm(proj)

class ImageProjModel(torch.nn.Module):
    """Projection model for reference image embeddings (same as IP-Adapter)"""
    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        proj = self.proj(image_embeds)  # (batch_size, clip_extra_context_tokens * cross_attention_dim)
        proj = proj.reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(proj)

class MultiSourceIPAttnProcessor(AttnProcessor):
    """Attention processor handling text, reference image, and body vector"""
    def __init__(self, hidden_size, cross_attention_dim, num_tokens_ref=4, num_tokens_body=4, lambda_ref=1.0, lambda_body=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens_ref = num_tokens_ref
        self.num_tokens_body = num_tokens_body
        self.lambda_ref = lambda_ref
        self.lambda_body = lambda_body
        self.has_ref = False

        # Projectors for reference image
        self.to_k_ref = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ref = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)

        # Projectors for body vector
        self.to_k_body = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_body = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def set_has_ref(self, has_ref: bool):
        self.has_ref = has_ref

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            attn_output = self._attention(query, key, value, attention_mask)
        else:
            # Split encoder_hidden_states into text, ref (if present), and body parts
            seq_len_text = 77  # Fixed length for CLIP text embeddings
            if self.has_ref:
                total_additional = self.num_tokens_ref + self.num_tokens_body
                text_end = seq_len_text
                ref_end = text_end + self.num_tokens_ref
                encoder_hidden_states_text = encoder_hidden_states[:, :text_end]
                encoder_hidden_states_ref = encoder_hidden_states[:, text_end:ref_end]
                encoder_hidden_states_body = encoder_hidden_states[:, ref_end:]
            else:
                total_additional = self.num_tokens_body
                text_end = seq_len_text
                encoder_hidden_states_text = encoder_hidden_states[:, :text_end]
                encoder_hidden_states_ref = None
                encoder_hidden_states_body = encoder_hidden_states[:, text_end:]

            # Standard text attention
            key_text = attn.to_k(encoder_hidden_states_text)
            value_text = attn.to_v(encoder_hidden_states_text)
            attn_output = self._attention(query, key_text, value_text, attention_mask)

            # Additional attention for reference image
            if self.has_ref and encoder_hidden_states_ref is not None:
                key_ref = self.to_k_ref(encoder_hidden_states_ref)
                value_ref = self.to_v_ref(encoder_hidden_states_ref)
                ref_attn = self._attention(query, key_ref, value_ref, None)
                attn_output = attn_output + self.lambda_ref * ref_attn

            # Additional attention for body vector
            key_body = self.to_k_body(encoder_hidden_states_body)
            value_body = self.to_v_body(encoder_hidden_states_body)
            body_attn = self._attention(query, key_body, value_body, None)
            attn_output = attn_output + self.lambda_body * body_attn

        attn_output = attn.to_out[0](attn_output)
        return attn_output

    def _attention(self, query, key, value, attention_mask):
        attn_weights = torch.bmm(query, key.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.bmm(attn_weights, value)

class BodyIPAdapter:
    """IP-Adapter extended with body vector conditioning"""
    def __init__(
        self,
        sd_pipe: StableDiffusionPipeline,
        image_encoder_path: str,
        ip_ckpt: Optional[str] = None,
        device: str = "cuda",
        num_tokens_ref: int = 4,
        num_tokens_body: int = 4,
        body_dim: int = 756,
    ):
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.num_tokens_ref = num_tokens_ref
        self.num_tokens_body = num_tokens_body

        # Initialize models
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_tokens_ref,
        ).to(self.device, dtype=torch.float16)
        self.body_proj_model = BodyProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            body_dim=body_dim,
            num_tokens=num_tokens_body,
        ).to(self.device, dtype=torch.float16)

        self.set_ip_adapter()
        if ip_ckpt:
            self.load_ip_adapter(ip_ckpt)

    def set_ip_adapter(self):
        """Configure UNet with MultiSourceIPAttnProcessor"""
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = unet.config.cross_attention_dim if not name.endswith("attn1.processor") else None
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):len("up_blocks.")+1])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):len("down_blocks.")+1])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = MultiSourceIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens_ref=self.num_tokens_ref,
                    num_tokens_body=self.num_tokens_body,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self, ip_ckpt: str):
        """Load pretrained weights"""
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def get_image_embeds(self, pil_image: Image.Image):
        """Generate embeddings for reference image"""
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image).image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def get_body_embeds(self, c_body: torch.Tensor):
        """Generate embeddings for body vector"""
        c_body = c_body.to(self.device, dtype=torch.float16)
        body_embeds = self.body_proj_model(c_body)
        uncond_body_embeds = self.body_proj_model(torch.zeros_like(c_body))
        return body_embeds, uncond_body_embeds

    def set_scales(self, lambda_ref: float, lambda_body: float, has_ref: bool):
        """Set attention scales and reference flag"""
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, MultiSourceIPAttnProcessor):
                attn_processor.lambda_ref = lambda_ref
                attn_processor.lambda_body = lambda_body
                attn_processor.set_has_ref(has_ref)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        c_body: torch.Tensor,
        ref_image: Optional[Image.Image] = None,
        negative_prompt: Optional[str] = None,
        lambda_ref: float = 1.0,
        lambda_body: float = 1.0,
        num_samples: int = 1,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs
    ) -> StableDiffusionPipelineOutput:
        """
        Generate images conditioned on text, body vector, and optional reference image.

        Args:
            prompt (str): Textual description τ_i.
            c_body (torch.Tensor): Body vector c_body of shape (batch_size, 756).
            ref_image (Optional[Image.Image]): Reference image I_ref, if provided.
            negative_prompt (Optional[str]): Negative prompt for classifier-free guidance.
            lambda_ref (float): Weight for reference image attention.
            lambda_body (float): Weight for body vector attention.
            num_samples (int): Number of samples to generate per prompt.
            seed (Optional[int]): Random seed for reproducibility.
            guidance_scale (float): CFG scale.
            num_inference_steps (int): Number of denoising steps.
            **kwargs: Additional arguments for the pipeline.

        Returns:
            StableDiffusionPipelineOutput: Generated images.
        """
        # Prepare prompts
        if not isinstance(prompt, List):
            prompt = [prompt] * num_samples
        if negative_prompt is None:
            negative_prompt = [""] * num_samples
        else:
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_samples

        # Set attention scales and reference flag
        self.set_scales(lambda_ref, lambda_body, ref_image is not None)

        # Encode text prompts
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        # Get body embeddings
        body_embeds, uncond_body_embeds = self.get_body_embeds(c_body)

        # Prepare combined embeddings
        if ref_image is not None:
            ref_embeds, uncond_ref_embeds = self.get_image_embeds(ref_image)
            prompt_embeds = torch.cat([prompt_embeds, ref_embeds, body_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_ref_embeds, uncond_body_embeds], dim=1)
        else:
            prompt_embeds = torch.cat([prompt_embeds, body_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_body_embeds], dim=1)

        # Generate images
        generator = get_generator(seed, self.device)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs
        )
        return images