from pathlib import Path


def rename_files(directory, newname):
    dir_path = Path(directory)
    # Получаем отсортированный список файлов в директории
    files = [f for f in dir_path.iterdir() if f.is_file()]

    for index, file in enumerate(files, start=1):
        new_file = file.with_name(f"{newname}{index}{file.suffix}")
        file.rename(new_file)
        # print(f"Файл {file.name} переименован в {new_file.name}")


if __name__ == "__main__":
    newname = "selenagomez"
    directory = f"./{newname}"
    rename_files(directory, newname)
