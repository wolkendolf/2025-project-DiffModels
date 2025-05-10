# import subprocess
import torch
import argparse
import json
import re
from pathlib import Path
from PIL import Image

class ImageProcessor:
    def initialize_ip_model(self):
        from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
        from ip_adapter import IPAdapter

        base_dir = Path(__file__).parent.parent.parent
        image_encoder_path = str(base_dir / "weights" / "IP-Adapter" / "models" / "image_encoder")  # Абсолютный путь
        
        if not (Path(image_encoder_path) / "config.json").exists():
            raise FileNotFoundError(
                f"Image encoder config not found at {image_encoder_path}. "
                "Please check the path or download the model."
            )
        
        base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        ip_ckpt = str(base_dir / "weights" / "IP-Adapter" / "models" / "ip-adapter_sd15.bin")
        device = "cuda:3"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        self.ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    def generate_images(self, data_dir, output_dir, path_to_image_and_text, num_images=4):
        """
        data_dir (str): dataset address
        output_dir (str): address where to put the generated images
        path_to_image_and_text (str): path to json file with picture description
        num_images (int): number of generated images
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        metadata_path = Path(path_to_image_and_text)
        
        # Читаем метаданные из jsonl файла
        with open(metadata_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                image_file = entry['image_file']
                prompt_text = entry['text']
                
                # Извлекаем имя персонажа из пути к изображению (например, "myleene_klass")
                person_name = re.match(r"(.+?)(\d+)\.jpg$", image_file).group(1)
                # Формируем полный путь к изображению
                image_path = data_dir / person_name / image_file
                
                # Открываем и подготавливаем изображение
                try:
                    image = Image.open(image_path)
                    image = image.resize((512, 512))
                    torch.cuda.empty_cache()
                    # Генерируем изображения с помощью IPAdapter
                    images = self.ip_model.generate(
                        pil_image=image,
                        num_samples=num_images,
                        num_inference_steps=50,
                        seed=42,
                        prompt=prompt_text,
                        scale=0.6
                    )
                    
                    # Создаем директорию для сохранения результатов
                    output_person_dir = output_dir / "ip-adapter_baseline"
                    output_person_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Сохраняем сгенерированные изображения
                    for i, img in enumerate(images):
                        img_path = output_person_dir / f'{person_name}_{i+1}.jpg'
                        img.save(img_path)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
                
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str,
                        help="Path to save the generated images")
    parser.add_argument("--json", type=str,
                        help="Path to json file with image description")
    parser.add_argument("--num_images", type=int,
                        help="Number of images to generate", default=4)
    args = parser.parse_args()

    ip_processor = ImageProcessor()
    # ip_processor.run_commands()
    ip_processor.initialize_ip_model()
    
    try:
        ip_processor.generate_images(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            path_to_image_and_text=args.json,
            num_images=args.num_images
        )
        print("Image generation completed successfully!")

    except Exception as e:
        print(f"An error occurred during image generation: {str(e)}")
        raise
