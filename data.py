import requests
import re


def download_and_save():
    print("Скачивание данных...")
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    try:
        raw_text = requests.get(url).text.lower()
        # Оставляем только буквы и пробелы
        clean_text = re.sub(r'[^a-z\s]', '', raw_text)

        with open("dataset.txt", "w", encoding="utf-8") as f:
            f.write(clean_text)
        print("Файл dataset.txt успешно сохранен!")
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")


if __name__ == "__main__":
    download_and_save()