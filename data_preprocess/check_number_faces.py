import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Путь к вашему датасету
BASE_DATASET_PATH = "/data/kazachkovda/2025_ipAdap_image"
model_path = "/home/kazachkovda/2025-project-DiffModels/weights/number_faces/yolo_nface.pt"

# Загрузка модели
if os.path.exists(model_path):
    print("Веса уже загружены.")
else:
    print("Скачиваем веса...")
    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt", local_dir="/home/kazachkovda/2025-project-DiffModels/weights/number_faces")
model = YOLO(model_path)

# Функция для проверки количества лиц на изображении
def check_faces(image_path):
    # Предсказание
    results = model.predict(image_path)
    # Получаем количество обнаруженных лиц (количество bounding boxes)
    num_faces = len(results[0].boxes)
    return num_faces

# Обход всех папок и файлов
for folder_name in os.listdir(BASE_DATASET_PATH):
    folder_path = os.path.join(BASE_DATASET_PATH, folder_name)
    
    # Проверяем, что это папка
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            # Проверяем, что это файл изображения
            if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Проверяем количество лиц
                    num_faces = check_faces(image_path)
                    
                    # Если больше 1 лица - удаляем файл
                    if num_faces > 1:
                        os.remove(image_path)
                        print(f"Удалено {image_path} - найдено {num_faces} лиц")
                    else:
                        print(f"Оставлено {image_path} - найдено {num_faces} лиц")
                        
                except Exception as e:
                    print(f"Ошибка при обработке {image_path}: {str(e)}")

print("Обработка завершена!")