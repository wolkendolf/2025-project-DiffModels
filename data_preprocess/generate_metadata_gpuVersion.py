import os

# Set the visible GPUs. Note that after this, GPU 0 in your process will be physical GPU 3.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,6"  # Put this at the very top!
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import json
import torch
from accelerate import Accelerator
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# These keys are the local device ids (0, 1, 2, 3).
max_memory = {
    0: "10GiB",
    1: "9GiB",
    2: "10GiB",
    3: "10GiB",
}
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
bf16_available = torch.cuda.is_bf16_supported()

print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Load the model using a GPU-friendly data type and let device_map auto-distribute layers across available GPUs.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_config,
    device_map={"": 0},  # Let Hugging Face distribute the layers
    torch_dtype=torch.bfloat16,  # Use torch.float16 for GPU compatibility
    max_memory=max_memory,
)

model.gradient_checkpointing_enable()

#check memory
print(f"Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

accelerator = Accelerator()

# Use Accelerator to prepare the model (this will also handle multi-GPU setup)
model = accelerator.prepare(model)

# Base directory with your images
base_dir = "/data/kazachkovda/2025_ipAdap_image"

# Gather image file paths
image_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(root, file))

# Partition the workload among processes if running multi-GPU distributed
if accelerator.num_processes > 1:
    image_files = image_files[accelerator.process_index :: accelerator.num_processes]

metadata = []

# Process each image
for image_path in image_files:
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 1024), Image.LANCZOS)
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Identify the sex of a person as woman or man. Describe this image with details on the person's body position, face, clothes, and a little background information."
                    ),
                },
            ],
        }
    ]

    # Prepare the input text and image tensors
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    file_name = os.path.basename(image_path)
    metadata.append({"image_file": file_name, "text": output_text})
    print(f"Processed {file_name}")

    # Очистка памяти
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

output_file = "metadata.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in metadata:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Metadata generation complete. Saved to {output_file}")
