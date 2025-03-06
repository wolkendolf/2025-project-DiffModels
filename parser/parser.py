import os
import re
import time
import parser.parsing_helper as parsing_helper
import requests
from bs4 import BeautifulSoup


def get_images_number(person_page_url, headers=None, delay=0):
    """
    Заходит на страницу конкретного человека, проверяет, сколько
    изображений доступно для скачивания.
    """
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

        # Если не нашли, возвращаем None или подставляем альтернативную логику.
        return None

    except Exception as e:
        print(f"Ошибка при загрузке страницы {person_page_url}: {e}")
        return None


def give_letters_pages(
    page_url,
    base_url,
    delay=0,
    headers=None,
):
    # Задержка перед запросом к странице (чтобы не "спамить" сервер)
    time.sleep(delay)

    try:
        resp = requests.get(page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 0. Составляем список знаменитостей по алфавиту
        letters_pages = []

        alphabet_block_latin = soup.select_one("div.letters-list")
        if alphabet_block_latin:
            # Находим все <a> внутри блока
            letters_lat = alphabet_block_latin.find_all("a", href=True)
            for letter_lat in letters_lat:
                letter_lat_page_url = letter_lat["href"]
                # Если ссылка относительная, делаем абсолютную
                if letter_lat_page_url[0] == "/":
                    letter_lat_page_url = base_url + letter_lat_page_url
                letters_pages.append(letter_lat_page_url)
        return letters_pages

    except Exception as e:
        print(f"Ошибка при загрузке {page_url}: {e}")


def parser_main(
    page_url,
    base_url,
    min_images=None,
    max_images=None,
    max_persons=None,
    delay=0.2,
    headers=None,
):
    try:
        total_visited = 0

        # 0. Составляем список знаменитостей по алфавиту
        letters_pages = give_letters_pages(
            page_url="https://www.theplace.ru/photos/",
            base_url=base_url,
            delay=0.1,
            headers=headers,
        )

        for letter_page in letters_pages:
            resp = requests.get(letter_page, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # 1. Ищем блок div.gallery-pics-list
            gallery_block = soup.select_one("div.models_list.row.my-5")
            if gallery_block:
                # Находим все <a> внутри блока
                persons_links = gallery_block.find_all("a", href=True)
                for a_tag in persons_links:
                    time.sleep(delay)
                    # Если достигли лимита скачиваний, завершаем
                    if max_persons is not None and total_visited > max_persons:
                        print("Достигнут лимит скачивания изображений.")
                        return total_visited

                    person_page_url = a_tag["href"]
                    # Если ссылка относительная, делаем абсолютную
                    if person_page_url.startswith("/"):
                        person_page_url = base_url + person_page_url

                    # Переходим на страницу личности, проверяем количество фотографий
                    enough_images = get_images_number(
                        person_page_url, headers=headers, delay=delay
                    )
                    if enough_images and enough_images >= min_images:
                        visited_pages = set()
                        parsing_helper.parse_page(
                            page_url=person_page_url,
                            base_url=base_url,
                            visited_pages=visited_pages,
                            folder=str(person_page_url.rstrip("/").split("/")[-1]),
                            max_images=max_images,
                            total_downloaded=0,
                            delay=delay,
                            headers=headers,
                        )
                        total_visited += 1

    except Exception as e:
        print(f"Ошибка при загрузке {page_url}: {e}")


def main(
    start_url,
    min_images,
    max_images,
    max_persons,
    delay=0,
    user_agent=None,
):
    """
    Главная функция:
    - start_url: начальный URL (например, https://www.theplace.ru/photos/tom_hardy/)
    - max_persons: ограничение на количество скачиваемых изображений (int или None)
    - delay: задержка в секундах между запросами (int или float)
    - user_agent: строка User-Agent (если нужна для обхода блокировок)
    """
    # Определим базовый URL, чтобы корректно строить абсолютные ссылки
    # Для theplace.ru: "https://www.theplace.ru"
    # Можно извлечь, например, через разбор start_url, но для упрощения укажем вручную.
    base_url = "https://www.theplace.ru"

    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    parser_main(
        page_url=start_url,
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
        max_images=500,
        max_persons=500,
        delay=0.2,
    )
