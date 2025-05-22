import torch
import torchvision.transforms as transforms
from PIL import Image
from yolo11_pose import YOLO11Pose
from vae_encoder import VAEEncoder
from gaussian_weithing import process_image_with_keypoints

def extract_c_body(image_paths, device='cuda'):
    """
    Выделяет вектор c_body из множества 100 изображений.

    Аргументы:
        image_paths (list): Список путей к 100 изображениям.
        device (str): Устройство для вычислений ('cuda' или 'cpu').

    Возвращает:
        torch.Tensor: Вектор c_body размером (756,).
    """
    # Проверка количества изображений
    if len(image_paths) != 100:
        raise ValueError("Ожидается ровно 100 изображений.")

    # Список для хранения взвешенных изображений
    weighted_images = []

    # Трансформация для изменения размера изображений
    resize_transform = transforms.Resize((1024, 1024))

    # Загрузка модели YOLO11-pose
    yolo_model = YOLO11Pose().to(device)
    yolo_model.eval()

    # Обработка каждого изображения
    for image_path in image_paths:
        # Загрузка и изменение размера изображения
        image = Image.open(image_path).convert('RGB')
        image = resize_transform(image)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # (1, C, 1024, 1024)

        # Извлечение ключевых точек с помощью YOLO11-pose
        with torch.no_grad():
            keypoints = yolo_model(image_tensor)  # Предполагается, что выход (1, 17, 2)

        # Преобразование ключевых точек в список кортежей (x, y)
        keypoints = keypoints.squeeze(0).cpu().numpy()  # (17, 2)
        keypoints_list = [
            (kp[0], kp[1]) if kp[0] > 0 and kp[1] > 0 else (0, 0)
            for kp in keypoints
        ]  # Зануление невидимых точек

        # Применение гауссова взвешивания
        weighted_image = process_image_with_keypoints(
            image_path, keypoints_list, sigma=50.0, alpha=0.1, device=device
        )
        weighted_images.append(weighted_image)

    # Объединение взвешенных изображений в один тензор
    combined_tensor = torch.stack(weighted_images, dim=0)  # (100, C, H, W)

    # Подготовка тензора для VAE-энкодера (пример: сглаживание)
    combined_tensor = combined_tensor.view(100, -1)  # (100, C*H*W)

    # Загрузка VAE-энкодера
    vae_encoder = VAEEncoder().to(device)
    vae_encoder.eval()

    # Получение вектора c_body
    with torch.no_grad():
        c_body = vae_encoder(combined_tensor)  # Предполагается, что выход (756,)

    return c_body

# Пример использования
# if __name__ == "__main__":
#     # Список путей к изображениям (должен содержать 100 элементов)
#     image_paths = [f'path/to/image{j}.jpg' for j in range(1, 101)]
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Извлечение вектора c_body
#     c_body = extract_c_body(image_paths, device)
#     print(f"Размер вектора c_body: {c_body.shape}")  # Ожидается torch.Size([756])