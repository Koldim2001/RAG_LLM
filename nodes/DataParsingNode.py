import logging
import requests
from bs4 import BeautifulSoup

from elements.DataElement import DataElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)

class DataParsingNode:
    """Модуль, отвечающий за парсинг веб-страниц."""
    
    def __init__(self, config) -> None:
        self.min_words = config["min_words"]

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
            
            # Если это GitHub-страница, применяем специальную фильтрацию
            if "github" in url.lower():
                logger.debug("Обнаружена ссылка на GitHub. Применяется специальная фильтрация.")
                final_text = DataParsingNode.filter_github_text(final_text)
            
            return {'text': final_text, 'description': description}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при загрузке страницы {url}: {e}")
            return {'text': '', 'description': url}

    @staticmethod
    def filter_github_text(text):
        """
        Удаляет нежелательные строки из текста GitHub-страницы.
        
        Args:
            text (str): Исходный текст.
        
        Returns:
            str: Отфильтрованный текст.
        """
        stop_phrases = [
            "Skip to content",
            "Write better code with AI",
            "Find and fix vulnerabilities",
            "Automate any workflow",
            "Instant dev environments",
            "Plan and track work",
            "Manage code changes",
            "Collaborate outside of code",
            "Find more, search less",
            "By company size",
            "Small and medium teams",
            "By use case",
            "View all use cases",
            "View all industries",
            "View all solutions",
            "White papers, Ebooks, Webinars",
            "Fund open source developers",
            "The ReadME Project",
            "GitHub community articles",
            "AI-powered developer platform",
            "Enterprise-grade security features",
            "Enterprise-grade AI features",
            "Enterprise-grade 24/7 support",
            "Search or jump to...",
            "Search code, repositories, users, issues, pull requests...",
            "Search syntax tips",
            "We read every piece of feedback, and take your input very seriously.",
            "Include my email address so I can be contacted",
            "Use saved searches to filter your results more quickly",
            "To see all available qualifiers, see our",
            "Create saved search",
            "You signed in with another tab or window.",
            "to refresh your session.",
            "You signed out in another tab or window.",
            "to refresh your session.",
            "You switched accounts on another tab or window.",
            "to refresh your session.",
            "Block or Report",
            "Prevent this user from interacting with your repositories and sending you notifications.",
            "You must be logged in to block users.",
            "Please don't include any personal information such as legal names or email addresses. Maximum 100 characters, markdown supported. This note will be visible to only you."
            "Contact GitHub support about this user’s behavior.",
            "You must be signed in to change notification settings",
            "Additional navigation options",
            "Go to file",
            "Folders and files",
            "Last commit message",
            "Last commit date",
            "View all files",
            "Repository files navigation",
        ]
        
        lines = text.split("\n")
        filtered_lines = [line for line in lines if not any(phrase in line for phrase in stop_phrases)]
        return "\n".join(filtered_lines)

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

    @profile_time
    def process(self, data_element: DataElement) -> DataElement:
        """
        Основной метод обработки данных.
        
        Args:
            data_element (DataElement): Элемент данных для обработки.
        
        Returns:
            DataElement: Обработанный элемент данных.
        """
        urls = data_element.url_list
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