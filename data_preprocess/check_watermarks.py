import os
import sys
from PIL import Image, ImageFile

# Позволяет Pillow загружать поврежденные изображения (иначе будет ошибка)
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(
    "/home/kazachkovda/2025-project-DiffModels/data_preprocess/watermark-detection"
)

from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor

"""
Необходимо установить зависимости из репозитория
https://github.com/boomb0om/watermark-detection/tree/main
Запуск из папки data_preprocess.
"""


# Базовый путь к датасету
BASE_DATASET_PATH = (
    "/data/kazachkovda/2025_ipAdap_image"
)
device = 'cuda:3'

# Функция для получения списка всех изображений
def get_all_images(base_path):
    image_paths = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):  # Поддерживаемые форматы
                    img_path = os.path.join(folder_path, img_file)
                    image_paths.append(img_path)
    return image_paths

# Фильтрация и удаление поврежденных изображений
def filter_valid_images(image_paths):
    valid_images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()  # Быстрая проверка файла
            valid_images.append(path)
        except Exception as e:
            print(f"[Удалено поврежденное] {path} — ошибка: {e}")
            try:
                os.remove(path)
            except Exception as delete_err:
                print(f"[Ошибка удаления] {path}: {delete_err}")
    return valid_images

# Инициализация модели
model, transforms = get_watermarks_detection_model(
    "convnext-tiny",
    device=device,
    fp16=False,
    cache_dir="/home/kazachkovda/2025-project-DiffModels/weights/watermarks",  # Укажите путь для кэша весов
)
predictor = WatermarksPredictor(
    model, transforms, device
)

# Получаем список всех изображений
image_paths = get_all_images(BASE_DATASET_PATH)
print(f"Найдено {len(image_paths)} изображений перед фильтрацией")

image_paths = filter_valid_images(image_paths)
print(f"Осталось {len(image_paths)} изображений после удаления поврежденных")

# Проверка на наличие водяных знаков
results = predictor.run(
    image_paths, num_workers=4, bs=64
)  # bs - размер батча, num_workers - кол-во потоков

# Удаление изображений с водяными знаками
for img_path, result in zip(image_paths, results):
    if result:  # True означает, что найден водяной знак
        print(f"[Водяной знак найден] {img_path}, удаляем...")
        try:
            os.remove(img_path)
            print(f"[Удалено] {img_path}")
        except Exception as e:
            print(f"[Ошибка при удалении] {img_path}: {e}")
    else:
        print(f"[Чисто] {img_path}")

print("Проверка завершена!")
