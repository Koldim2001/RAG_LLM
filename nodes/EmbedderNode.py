import logging
import requests
from typing import List
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class EmbedderNode:
    """Модуль отвечающий, за создание эмбеддинга текстов"""

    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        
        self.embedder_url = f"http://{self.host}:{self.port}/embed"

    @profile_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получает эмбеддинги для списка текстов."""
        response = requests.post(
            self.embedder_url,
            json={"inputs": texts}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Ошибка: {response.status_code}, {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """Получает эмбеддинг для одного текста."""
        return self.embed_documents([text])[0]