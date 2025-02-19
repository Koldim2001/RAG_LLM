import logging
from typing import List
import requests
from langchain.schema import Document

from elements.QueryElement import QueryElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class RerankerNode:
    """Модуль отвечающий, за реранжирование текстов"""

    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.reranker_url = f"http://{self.host}:{self.port}/rerank"
        self.min_score = config["min_score"]
        self.top_k = config["top_k"]
 
    @profile_time
    def process(self, query_element: QueryElement) -> QueryElement:
        top_chunks = query_element.top_chunks
        reranked_docs = self.rerank(query=query_element.upgraded_query, documents=top_chunks)
        query_element.top_chunks = reranked_docs

        # Фильтрация топ-документов по условию score > self.min_score
        filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("score_rerank", 0) > self.min_score]
        query_element.prompt_chunks = filtered_docs[:self.top_k]
            
        return query_element

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Пересчитывает релевантность документов на основе запроса.
        """
        # Преобразуем документы в текстовый формат
        texts = [doc.page_content for doc in documents]
        # Отправляем запрос к реранкеру
        response = requests.post(
            self.reranker_url,
            json={"query": query, "texts": texts}
        )
        if response.status_code != 200:
            raise Exception(f"Ошибка: {response.status_code}, {response.text}")
        # Получаем результаты реранкинга
        results = response.json()
        reranked_docs = []
        # Сортируем результаты по убыванию оценки
        results.sort(key=lambda x: x["score"], reverse=True)
        # Сопоставляем результаты с документами
        for result in results:
            index = result["index"]
            score = result["score"]
            doc = documents[index]
            doc.metadata["score_rerank"] = score
            reranked_docs.append(doc)
        return reranked_docs
