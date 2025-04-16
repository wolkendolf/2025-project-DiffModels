import os
import shutil

"""
При необходимости проверить, что картинок в каждой папке не меньше какого-то порога
"""

BASE_DATASET_PATH = "/home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/dataset"

delete_count_folder = 0
total_count_folder = 0
count_images = 0
# Получаем список всех папок в BASE_DATASET_PATH
for folder_name in os.listdir(BASE_DATASET_PATH):
    folder_path = os.path.join(BASE_DATASET_PATH, folder_name)
    if os.path.isdir(folder_path):
        total_count_folder += 1
        # Считаем количество изображений (файлов с расширением .jpg)
        num_images = len(
            [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith(".jpg")
            ]
        )
        count_images += num_images
        if num_images < 100:
            delete_count_folder += 1
            print(f"{folder_name}: {num_images} изображений")
            # shutil.rmtree(folder_path)

print(
    f"Удалено {delete_count_folder} папок. Осталось {total_count_folder - delete_count_folder}."
)
print(f"Всего изображений {count_images}.")
