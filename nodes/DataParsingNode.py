import logging
import requests
from bs4 import BeautifulSoup

from elements.DataElement import DataElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class DataParsingNode:
    """Модуль отвечающий, за парсинг веб страниц"""

    def __init__(self, config) -> None:
        self.min_words = config["min_words"]
        
    @profile_time
    def process(self, data_element: DataElement) -> DataElement:
        urls = data_element.url_list

        # Парсинг и фильтрация
        url_data = {}
        for url in urls:
            logger.info(f"Парсинг {url}...")
            parsed_data = self.parse_url(url)
            if parsed_data['text']:
                filtered_text = self.filter_text(parsed_data['text'])
                url_data[url] = {
                    'text': filtered_text,
                    'description': parsed_data['description']
                    }
        data_element.url_data = url_data  

        return data_element

    @staticmethod
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

    @staticmethod
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
        filtered_text = [line.strip() for line in lines if len(line.split()) >= min_words]

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
