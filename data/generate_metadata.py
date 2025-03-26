import os
import json
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

accelerator = Accelerator(cpu=True)

# Load the model and processor (adjust torch_dtype and device_map as needed)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

# Move model to the accelerator (it will automatically use all available GPUs)
model = accelerator.prepare(model)

# Ensure the model is on the correct device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Base directory containing your images
base_dir = "/data/kazachkovda/popstars/train"

# Gather all image file paths (you can extend the extensions if needed)
image_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(root, file))

# This list will store metadata entries
metadata = []

# Process each image one by one
for image_path in image_files:
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        continue

    # Prepare the message structure with a custom prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": ("Describe this image with details on the person's body position, face, clothes, "
                             "and a little background information.")
                },
            ],
        }
    ]

    # Apply the chat template and process visual inputs
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate description for the image
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    # Trim the generated output to remove the prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    file_name = os.path.basename(image_path)
    entry = {
        "file_name": file_name,
        "text": output_text
    }
    metadata.append(entry)
    print(f"Processed {file_name}")

# Write the metadata to a JSON Lines file (one JSON object per line)
output_file = "metadata.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in metadata:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Metadata generation complete. Saved to {output_file}")