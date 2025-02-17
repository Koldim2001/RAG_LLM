
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_url(url):
    """
    Парсит содержимое указанного URL и возвращает текстовое содержимое страницы.

    Args:
        url (str): URL-адрес для парсинга.

    Returns:
        dict: Словарь с ключами 'text' (текстовая информация) и 'description' (краткое описание страницы).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Извлекаем краткое описание (title страницы)
        description = soup.title.string.strip() if soup.title else url

        # Обработка контента
        soup_copy = BeautifulSoup(response.text, "html.parser")
        for tag in soup_copy.find_all(['pre', 'code']):
            tag.replace_with(tag.get_text(separator=" ", strip=True))
        final_text = soup_copy.get_text(separator="\n", strip=True)

        return {'text': final_text, 'description': description}

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке страницы {url}: {e}")
        return {'text': '', 'description': url}


def filter_text(text, min_words=3):
    """
    Фильтрует текст, удаляя строки с минимальным количеством слов и стоп-фразы.

    Args:
        text (str): Исходный текст.
        min_words (int): Минимальное количество слов в строке для сохранения.

    Returns:
        str: Отфильтрованный текст.
    """
    if not text:
        return ''

    lines = text.split("\n")
    filtered_text = [line.strip() for line in lines if len(line.split()) > min_words]

    # Поиск стоп-фраз
    stop_phrases = [
        "Политика в отношении файлов cookie",
        "Мы используем cookie",
        "Дата обращения:",
        "Использованная литература и источники:"
    ]
    cutoff_index = next((i for i, line in enumerate(filtered_text) if any(phrase in line for phrase in stop_phrases)), None)

    if cutoff_index is not None:
        filtered_text = filtered_text[:cutoff_index]

    return "\n".join(filtered_text)


def generate_chunks(url_data, chunk_size=1500, chunk_overlap=0):
    """
    Генерирует чанки из текстов, добавляя описание источника в начало каждого чанка.

    Args:
        url_data (dict): Словарь с текстами и описаниями страниц.
        chunk_size (int): Максимальный размер чанка.
        chunk_overlap (int): Перекрытие между чанками.

    Returns:
        list: Список отформатированных чанков.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    all_chunks = []
    for url, data in url_data.items():
        chunks = text_splitter.split_text(data['text'])
        for chunk in chunks:
            # Добавляем описание в начало чанка
            formatted_chunk = f"[Источник: {data['description']}]\n{chunk}"
            all_chunks.append(formatted_chunk)

    return all_chunks


# Список URL-адресов
urls = [
    "https://www.eurochem.ru/",
    "https://www.eurochem.ru/global-operations/",
    "https://www.eurochem.ru/about-us/komplaens/",
    "https://www.eurochem.ru/proteh-lab/",
    "https://digtp.com/",
    "https://digtp.com/projects/machine-learning-platforma",
    "https://digtp.com/projects/rekomendatelnye-modeli",
    "https://digtp.com/projects/mobilnoe-prilozenie-mineralogiia",
    "https://digtp.com/contacts",
    "https://www.eurochem-career.com/news/iskusstvennyi-intellekt-v-ximii-gpt-assistenty-v-evroxime",
    "https://otus.ru/instructors/10517",
    "https://ru.wikipedia.org/wiki/ЕвроХим",
    "https://www.eurochem.ru/usolskij-kalijnyj-kombinat/",
    "https://uralmines.ru/evrohim-usolskij-kalijnyj-kombinat/",
    "https://docs.ultralytics.com/tasks/segment",
    "https://docs.ultralytics.com/tasks/detect",
    "https://docs.ultralytics.com/tasks",
    "https://docs.ultralytics.com/modes/",
    "https://docs.ultralytics.com/solutions",
    "https://github.com/Koldim2001",
    "https://github.com/Koldim2001/YOLO-Patch-Based-Inference",
    "https://github.com/Koldim2001/TrafficAnalyzer",
    "https://github.com/Koldim2001/COCO_to_YOLOv8"
]

# Парсинг и фильтрация
url_data = {}
for url in urls:
    print(f"Парсинг {url}...")
    parsed_data = parse_url(url)
    if parsed_data['text']:
        filtered_text = filter_text(parsed_data['text'])
        url_data[url] = {
            'text': filtered_text,
            'description': parsed_data['description']
        }


# Генерация чанков через функцию
all_chunks = generate_chunks(url_data, chunk_size=1250, chunk_overlap=0)

# Сохранение чанков
with open("chunks_output.txt", "w", encoding="utf-8") as file:
    for i, chunk in enumerate(all_chunks):
        file.write(f"Чанк {i+1} ({len(chunk)} символов):\n{chunk}\n{'='*50}\n")
print("Чанки сохранены в файл chunks_output.txt")