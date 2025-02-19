import logging
from utils_local.utils import profile_time
import yaml

from elements.QueryElement import QueryElement
from nodes.RerankerNode import RerankerNode
from nodes.VectorDBNode import VectorDBNode
from nodes.EmbedderNode import EmbedderNode
from nodes.LLMNode import LLMNode

# Настройка логгера для работы в режиме INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class AskLLM:
    """Модуль отвечающий, за обзение с LLM с использованием RAG и без него"""

    def __init__(self) -> None:
        with open("configs/app_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        self.vector_db_node = VectorDBNode(config["vector_db_node"])
        self.embedder_node = EmbedderNode(config["embedder_node"])
        self.reranker_node = RerankerNode(config["reranker_node"])
        self.llm_node = LLMNode(config["llm_node"])
       
    @profile_time
    def process(self, query: str, message_number=0, collection_db_name=None, previous_messages=[], show_data_info=False):
        """Наполняет датасет по данным url"""
        query_element = QueryElement(query, message_number, collection_db_name, previous_messages)
        # уточнение запроса с учетом чата:
        query_element.upgraded_query = self.llm_node.make_abstract(query_element)

        if collection_db_name is not None and self.vector_db_node.db_has_collection(collection_db_name):
            embedding = self.embedder_node.embed_query(query_element.upgraded_query)
            similar_chunks = self.vector_db_node.search_similar_chunks(collection_db_name, embedding)
            query_element.top_chunks = self.embedder_node.chunks_to_documents(similar_chunks)

            if show_data_info:
                print("Результат векторного поиска:")
                query_element.display_chunks(query_element.top_chunks, limit_size=200)
                print("\n=======================\n")
            
            query_element = self.reranker_node.process(query_element)

            if show_data_info:
                print("Результат после реранка:")
                query_element.display_chunks(query_element.prompt_chunks, limit_size=200)
                print("\n=======================\n")

            query_element = self.llm_node.answer_with_rag(query_element, show_data_info)
        else:
            query_element = self.llm_node.answer(query_element, show_data_info)

        return query_element
    
    def get_new_history(self, query_element: QueryElement) -> list:
        return self.llm_node.get_new_history(query_element)