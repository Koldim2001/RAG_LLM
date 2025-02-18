import logging
import requests
from typing import List
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)

class EmbedderNode:
    """Модуль, отвечающий за создание эмбеддинга текстов."""
    
    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.batch_size = config["max_batch_size"]  # Размер батча, можно настроить через конфиг
        
        self.embedder_url = f"http://{self.host}:{self.port}/embed"
    
    @profile_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов, разбивая их на батчи.
        
        Args:
            texts (List[str]): Список текстов для создания эмбеддингов.
        
        Returns:
            List[List[float]]: Список эмбеддингов для каждого текста.
        """
        embeddings = []
        
        # Разбиваем тексты на батчи заданного размера
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(f"Обработка батча {i // self.batch_size + 1}, размер батча: {len(batch_texts)}")
            
            # Отправляем запрос для текущего батча
            response = requests.post(
                self.embedder_url,
                json={"inputs": batch_texts}
            )
            
            if response.status_code == 200:
                batch_embeddings = response.json()
                embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"Ошибка при обработке батча: {response.status_code}, {response.text}")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для одного текста.
        
        Args:
            text (str): Текст для создания эмбеддинга.
        
        Returns:
            List[float]: Эмбеддинг для указанного текста.
        """
        return self.embed_documents([text])[0]