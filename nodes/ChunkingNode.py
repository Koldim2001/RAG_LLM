import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elements.DataElement import DataElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class ChunkingNode:
    """Модуль отвечающий, за парсинг веб страниц"""

    def __init__(self, config) -> None:
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.separators = ["\n\n", "\n", ". ", " "]
        
    @profile_time
    def process(self, data_element: DataElement) -> DataElement:
        url_data = data_element.url_data

        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        all_chunks = []
        for url, data in url_data.items():
            chunks = text_splitter.split_text(data['text'])
            for chunk in chunks:
                # Добавляем описание в начало чанка
                formatted_chunk = f"[Источник: {data['description']}]\n{chunk}"
                all_chunks.append(formatted_chunk)

        data_element.chunks = all_chunks
        return data_element
