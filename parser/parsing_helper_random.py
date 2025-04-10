import os
import time
import requests
from bs4 import BeautifulSoup


def download_image(img_url, folder, subfolder_name, idx):
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{subfolder_name}{idx}.jpg"
    filepath = os.path.join(folder, filename)

    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Ошибка при скачивании {img_url}: {e}")
        return False


def get_fullsize_image_url(image_page_url, headers=None, delay=0):
    time.sleep(delay)
    try:
        resp = requests.get(image_page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        img_tag = soup.select_one(
            "#vue-app > div.content-wrapper > div:nth-child(3) > div > div > "
            "div:nth-child(1) > div > div.pic-big-box__main > span.s-pic > img"
        )
        if img_tag and img_tag.get("src"):
            return img_tag["src"]
        return None
    except Exception as e:
        print(f"Ошибка при загрузке страницы {image_page_url}: {e}")
        return None


def parse_page(
    page_url,
    base_url,
    visited_pages,
    folder,
    subfolder_name,
    max_images=None,
    total_downloaded=0,
    delay=0,
    headers=None,
):
    if page_url in visited_pages:
        return total_downloaded

    visited_pages.add(page_url)
    print(f"Парсим страницу: {page_url}")
    time.sleep(delay)

    try:
        resp = requests.get(page_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        gallery_block = soup.select_one("div.row.gx-1.justify-content-center")
        if gallery_block:
            image_links = gallery_block.find_all("a", href=True)
            existing_files = (
                [
                    f
                    for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f))
                ]
                if os.path.exists(folder)
                else []
            )
            start_idx = len(existing_files)

            for idx, a_tag in enumerate(image_links, start=start_idx):
                if max_images is not None and total_downloaded >= max_images:
                    print("Достигнут лимит скачивания изображений для персоны.")
                    return total_downloaded

                image_page_url = a_tag["href"]
                if image_page_url.startswith("/"):
                    image_page_url = base_url + image_page_url

                fullsize_url = get_fullsize_image_url(
                    image_page_url, headers=headers, delay=delay
                )
                if fullsize_url:
                    if fullsize_url.startswith("/"):
                        fullsize_url = base_url + fullsize_url
                    time.sleep(delay)
                    success = download_image(fullsize_url, folder, subfolder_name, idx)
                    if success:
                        total_downloaded += 1

        if max_images is None or total_downloaded < max_images:
            listalka_block = soup.select_one("div.listalka")
            if listalka_block:
                page_links = listalka_block.find_all("a", href=True)
                for link in page_links:
                    if max_images is not None and total_downloaded >= max_images:
                        print("Достигнут лимит скачивания изображений.")
                        return total_downloaded

                    href = link["href"]
                    next_page_url = base_url + href if href.startswith("/") else href
                    total_downloaded = parse_page(
                        next_page_url,
                        base_url,
                        visited_pages,
                        folder=folder,
                        subfolder_name=subfolder_name,
                        max_images=max_images,
                        total_downloaded=total_downloaded,
                        delay=delay,
                        headers=headers,
                    )
                    if max_images is not None and total_downloaded >= max_images:
                        break

    except Exception as e:
        print(f"Ошибка при загрузке {page_url}: {e}")

    return total_downloaded
