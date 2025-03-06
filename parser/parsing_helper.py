import os
import re
import time
import requests
from bs4 import BeautifulSoup


def download_image(img_url, folder="images"):
    """
    Скачивает изображение по прямой ссылке img_url и сохраняет в папку folder.
    Возвращает True, если скачивание прошло успешно, иначе False.
    """
    # Создаём папку, если её нет
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Пытаемся извлечь имя файла из ссылки
    filename = img_url.split("/")[-1]
    # Если оно пустое или слишком короткое, можно придумать своё
    if not filename or "." not in filename:
        filename = f"{folder}{int(time.time())}.jpg"

    filepath = os.path.join(folder, filename)

    # Скачиваем файл
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # print(f"Скачан файл: {filepath}")
        return True
    except Exception as e:
        print(f"Ошибка при скачивании {img_url}: {e}")
        return False


def get_fullsize_image_url(image_page_url, headers=None, delay=0):
    """
    Заходит на страницу конкретного изображения (image_page_url),
    ищет тег <img> по заданному пути (JS-path), и возвращает его src.
    """
    import time
    from bs4 import BeautifulSoup
    import requests

    # Задержка (если нужно), чтобы снизить риск бана
    time.sleep(delay)

    try:
        resp = requests.get(image_page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Путь в CSS-селекторах (соответствует JS-path из вопроса):
        # #vue-app > div.content-wrapper > div:nth-child(3) > div > div > div:nth-child(1) >
        # div > div.pic-big-box__main > span.s-pic > img
        img_tag = soup.select_one(
            "#vue-app > div.content-wrapper > div:nth-child(3) > div > div > "
            "div:nth-child(1) > div > div.pic-big-box__main > span.s-pic > img"
        )

        # Если нашли <img> и у него есть src — возвращаем
        if img_tag and img_tag.get("src"):
            return img_tag["src"]

        # Если не нашли, возвращаем None или подставляем альтернативную логику.
        return None

    except Exception as e:
        print(f"Ошибка при загрузке страницы {image_page_url}: {e}")
        return None


def parse_page(
    page_url,
    base_url,
    visited_pages,
    folder="images",
    max_images=None,
    total_downloaded=0,
    delay=0,
    headers=None,
):
    """
    Парсит указанную страницу:
    1) Ищет все ссылки на блоке div.gallery-pics-list и скачивает изображения.
    2) Ищет блок div.listalka и извлекает ссылки пагинации, рекурсивно обходит их.

    Параметры:
    - page_url: текущая страница для парсинга
    - base_url: базовый URL, например https://www.theplace.ru
    - visited_pages: множество (set) уже посещённых страниц
    - folder: папка для сохранения изображений
    - max_images: максимально допустимое количество изображений для скачивания (int или None)
    - total_downloaded: текущее количество скачанных изображений (int)
    - delay: задержка в секундах перед запросами (float или int)
    - headers: словарь с заголовками (например, {'User-Agent': '...'})

    Возвращает количество скачанных изображений (total_downloaded).
    """
    if page_url in visited_pages:
        return total_downloaded  # Уже были на этой странице

    visited_pages.add(page_url)
    print(f"Парсим страницу: {page_url}")

    # Задержка перед запросом к странице (чтобы не "спамить" сервер)
    time.sleep(delay)

    try:
        resp = requests.get(page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Ищем блок div.gallery-pics-list
        gallery_block = soup.select_one("div.row.gx-1.justify-content-center")
        if gallery_block:
            # Находим все <a> внутри блока
            image_links = gallery_block.find_all("a", href=True)
            for a_tag in image_links:
                # Если достигли лимита скачиваний, завершаем
                if max_images is not None and total_downloaded >= max_images:
                    print("Достигнут лимит скачивания изображений.")
                    return total_downloaded

                image_page_url = a_tag["href"]
                # Если ссылка относительная, делаем абсолютную
                if image_page_url.startswith("/"):
                    image_page_url = base_url + image_page_url
                    # print(image_page_url)

                # Переходим на страницу картинки, получаем прямую ссылку и скачиваем
                fullsize_url = get_fullsize_image_url(
                    image_page_url, headers=headers, delay=delay
                )
                if fullsize_url:
                    # Если она относительная, склеим с базовым доменом
                    if fullsize_url.startswith("/"):
                        fullsize_url = base_url + fullsize_url

                    # Задержка перед скачиванием (можно объединить с общей задержкой выше)
                    time.sleep(delay)

                    success = download_image(fullsize_url, folder=folder)
                    if success:
                        total_downloaded += 1

        # 2. Ищем блок div.listalka для пагинации
        #    Если достигли лимита, не идём дальше
        if max_images is None or total_downloaded < max_images:
            listalka_block = soup.select_one("div.listalka")
            if listalka_block:
                page_links = listalka_block.find_all("a", href=True)
                for link in page_links:
                    #     # Если достигли лимита скачиваний, завершаем
                    if max_images is not None and total_downloaded >= max_images:
                        print("Достигнут лимит скачивания изображений.")
                        return total_downloaded

                    href = link["href"]
                    # Ищем ссылки вида "?page=n" или "/photos/tom_hardy/?page=n"
                    # Если ссылка относительная, приводим к абсолютной
                    if href.startswith("/"):
                        next_page_url = base_url + href
                    else:
                        next_page_url = href

                    # Рекурсивно обходим следующие страницы
                    total_downloaded = parse_page(
                        next_page_url,
                        base_url,
                        visited_pages,
                        folder=folder,
                        max_images=max_images,
                        total_downloaded=total_downloaded,
                        delay=delay,
                        headers=headers,
                    )
                    #     # Если достигли лимита, выходим из цикла
                    if max_images is not None and total_downloaded >= max_images:
                        break

    except Exception as e:
        print(f"Ошибка при загрузке {page_url}: {e}")

    return total_downloaded


# def parser_photo(start_url, folder="images", max_images=None, delay=0, user_agent=None):
#     """
#     Главная функция:
#     - start_url: начальный URL (например, https://www.theplace.ru/photos/tom_hardy/)
#     - folder: папка для сохранения
#     - max_images: ограничение на количество скачиваемых изображений (int или None)
#     - delay: задержка в секундах между запросами (int или float)
#     - user_agent: строка User-Agent (если нужна для обхода блокировок)
#     """
#     # Определим базовый URL, чтобы корректно строить абсолютные ссылки
#     # Для theplace.ru: "https://www.theplace.ru"
#     # Можно извлечь, например, через разбор start_url, но для упрощения укажем вручную.
#     base_url = "https://www.theplace.ru"

#     headers = {}
#     if user_agent:
#         headers["User-Agent"] = user_agent

#     visited_pages = set()
#     total_downloaded = parse_page(
#         page_url=start_url,
#         base_url=base_url,
#         visited_pages=visited_pages,
#         folder=folder,
#         max_images=max_images,
#         total_downloaded=0,
#         delay=delay,
#         headers=headers,
#     )

#     print(f"Всего скачано изображений: {total_downloaded}")


# if __name__ == "__main__":
#     # Пример запуска:
#     # Скачаем не более 10 изображений, задержка 2 секунды, пользовательский User-Agent
#     start_url = "https://www.theplace.ru/photos/selena_gomez/"
#     main(
#         start_url,
#         folder="selenagomez",
#         max_images=2000,
#         delay=0.2,
#     )
