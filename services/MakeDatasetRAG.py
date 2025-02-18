import logging
from typing import List
from utils_local.utils import profile_time
import yaml

from elements.DataElement import DataElement
from nodes.DataParsingNode import DataParsingNode
from nodes.ChunkingNode import ChunkingNode
from nodes.VectorDBNode import VectorDBNode
from nodes.EmbedderNode import EmbedderNode

# Настройка логгера для работы в режиме INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MakeDatasetRAG:
    """Модуль отвечающий, за создание датасета RAG на основании списка url"""

    def __init__(self) -> None:
        with open("configs/app_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        self.data_parsing_node = DataParsingNode(config["data_parsing_node"])
        self.chunking_node = ChunkingNode(config["chunking_node"])
        self.vector_db_node = VectorDBNode(config["vector_db_node"])
        self.embedder_node = EmbedderNode(config["embedder_node"])
       

    @profile_time
    def process(self, url_list: List[str], collection_db_name="default", show_data_info=False):
        """Наполняет датасет по данным url"""
        data_element = DataElement(url_list, collection_db_name)
        data_element = self.data_parsing_node.process(data_element)
        data_element = self.chunking_node.process(data_element)

        if show_data_info:
            data_element.save_chunks()
            data_element.save_parsing_result()
            print(data_element)

        embeddings = self.embedder_node.embed_documents(data_element.chunks)
        
        self.vector_db_node.delete_milvus_collection(collection_db_name)
        self.vector_db_node.create_milvus_collection(collection_db_name)
        self.vector_db_node.insert_data_into_milvus(collection_db_name, data_element.chunks, embeddings)
