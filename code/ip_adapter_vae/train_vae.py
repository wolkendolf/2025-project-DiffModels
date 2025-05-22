import os
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .utils import get_generator
from .attention_processor import AttnProcessor
from ip_adapter.ip_adapter import BodyIPAdapter

# Initialize pipeline
sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
adapter = BodyIPAdapter(
    sd_pipe=sd_pipe,
    image_encoder_path="openai/clip-vit-large-patch14",
    device="cuda",
)

# Sample inputs
prompt = ""
c_body = torch.randn(1, 756)  # body vector
ref_image = Image.open(".jpg").convert("RGB")

# Generate with reference image
images_with_ref = adapter.generate(
    prompt=prompt,
    c_body=c_body,
    ref_image=ref_image,
    lambda_ref=0.8,
    lambda_body=1.0,
    num_samples=1,
    seed=42,
)

# Generate without reference image
images_without_ref = adapter.generate(
    prompt=prompt,
    c_body=c_body,
    lambda_ref=0.0,  # Ignored since ref_image is None
    lambda_body=1.0,
    num_samples=1,
    seed=42,
)

# Save or display images
images_with_ref.images[0].save("output_with_ref.png")
images_without_ref.images[0].save("output_without_ref.png")