import os
import time
import parser.parsing_helper_random as parsing_helper
import requests
from bs4 import BeautifulSoup
import random

BASE_DATASET_PATH = (
    "/home/jovyan/home/jovyan/kazachkovda/2025-project-DiffModels/dataset"
)


def get_images_number(person_page_url, headers=None, delay=0):
    time.sleep(delay)
    try:
        resp = requests.get(person_page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        description_tag = soup.select_one(
            "#vue-app > div.content-wrapper > div:nth-child(2) > div > div.col-xl-9.col-lg-8.col-md-7.col-sm-12.main-col-content > div.d-none.d-sm-block"
        )
        if description_tag:
            p_tag = description_tag.find("p")
            return int(p_tag.get_text(strip=True).split(" ")[1])
        return None
    except Exception as e:
        print(f"Ошибка при загрузке страницы {person_page_url}: {e}")
        return None


def get_all_persons(base_url, headers=None, delay=0):
    """Собирает список всех персон с сайта"""
    persons_list = []
    try:
        # Сначала получаем все страницы букв
        resp = requests.get("https://www.theplace.ru/photos/", headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        alphabet_block = soup.select_one("div.letters-list")
        if alphabet_block:
            letter_links = alphabet_block.find_all("a", href=True)

            for letter_link in letter_links:
                time.sleep(delay)
                letter_url = letter_link["href"]
                if letter_url.startswith("/"):
                    letter_url = base_url + letter_url

                # Получаем страницу с людьми для каждой буквы
                letter_resp = requests.get(letter_url, headers=headers)
                letter_resp.raise_for_status()
                letter_soup = BeautifulSoup(letter_resp.text, "html.parser")

                gallery_block = letter_soup.select_one("div.models_list.row.my-5")
                if gallery_block:
                    person_links = gallery_block.find_all("a", href=True)
                    for person_link in person_links:
                        person_url = person_link["href"]
                        if person_url.startswith("/"):
                            person_url = base_url + person_url
                        persons_list.append(person_url)

        return persons_list

    except Exception as e:
        print(f"Ошибка при сборе списка персон: {e}")
        return []


def parser_main(
    base_url,
    min_images=None,
    max_images=None,
    max_persons=None,
    delay=0.2,
    headers=None,
):
    try:
        total_visited = 0
        # Получаем список всех персон
        all_persons = get_all_persons(base_url, headers=headers, delay=delay)

        if not all_persons:
            print("Не удалось получить список персон")
            return

        # Пока не достигли лимита персон или есть кого обрабатывать
        while all_persons and (max_persons is None or total_visited < max_persons):
            # Случайный выбор персоны
            person_page_url = random.choice(all_persons)
            all_persons.remove(person_page_url)  # Удаляем обработанную персону

            time.sleep(delay)

            subfolder_name = person_page_url.rstrip("/").split("/")[-1]
            folder_path = os.path.join(BASE_DATASET_PATH, subfolder_name)

            # Проверка существующей папки и количества файлов
            if os.path.exists(folder_path):
                existing_files = len(
                    [
                        f
                        for f in os.listdir(folder_path)
                        if os.path.isfile(os.path.join(folder_path, f))
                    ]
                )
                if existing_files >= max_images:
                    print(f"Пропускаем {subfolder_name}: уже скачано максимум фото")
                    continue

            enough_images = get_images_number(
                person_page_url, headers=headers, delay=delay
            )
            if enough_images and enough_images >= min_images:
                visited_pages = set()
                parsing_helper.parse_page(
                    page_url=person_page_url,
                    base_url=base_url,
                    visited_pages=visited_pages,
                    folder=folder_path,
                    subfolder_name=subfolder_name,
                    max_images=max_images,
                    total_downloaded=0,
                    delay=delay,
                    headers=headers,
                )
                total_visited += 1
                print(f"Обработано персон: {total_visited}")

    except Exception as e:
        print(f"Ошибка в основном процессе: {e}")


def main(
    start_url,
    min_images,
    max_images,
    max_persons,
    delay=0,
    user_agent=None,
):
    base_url = "https://www.theplace.ru"
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    if not os.path.exists(BASE_DATASET_PATH):
        os.makedirs(BASE_DATASET_PATH)

    parser_main(
        base_url=base_url,
        min_images=min_images,
        max_images=max_images,
        max_persons=max_persons,
        delay=delay,
        headers=headers,
    )


if __name__ == "__main__":
    start_url = "https://www.theplace.ru/photos/"
    main(
        start_url,
        min_images=200,
        max_images=600,
        max_persons=500,
        delay=0.2,
    )
