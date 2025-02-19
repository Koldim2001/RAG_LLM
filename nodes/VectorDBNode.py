import logging
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

logger = logging.getLogger(__name__)

class VectorDBNode:
    """Модуль отвечающий за работу с векторной базой данных"""
    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.dim = config["dim"]
        self.top_k = config["top_k"]

        # Создание подключения к Milvus
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Connected to Milvus at {self.host}:{self.port} successfully!")

    def db_has_collection(self, collection_name):
        # проверка наличия коллекции
        return utility.has_collection(collection_name)

    def create_milvus_collection(self, collection_name):
        """
        Создает новую коллекцию в Milvus, если она еще не существует.
        
        Args:
            collection_name (str): Имя коллекции.
        """
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return
        
        if self.dim <= 0:
            logger.error("Dimension must be greater than 0.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_length", dtype=DataType.INT64),
            FieldSchema(name="time_insert", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="Collection for text chunks")
        
        try:
            Collection(name=collection_name, schema=schema)
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
        
        self.create_index(collection_name)

    def create_index(self, collection_name, field_name="embedding", index_params=None):
        """
        Создает индекс на указанном поле в коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            field_name (str): Поле, на котором создается индекс (по умолчанию "embedding").
            index_params (dict): Параметры индекса (по умолчанию IVF_FLAT).
        """
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return

        collection = Collection(collection_name)
        if index_params is None:
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

        try:
            collection.create_index(field_name=field_name, index_params=index_params)
            logger.info(f"Index created on field '{field_name}' for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to create index on field '{field_name}' for collection '{collection_name}': {e}")

        try:
            collection.load()
            logger.info(f"Collection '{collection_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load collection '{collection_name}': {e}")

    def delete_milvus_collection(self, collection_name):
        """
        Удаляет коллекцию из Milvus, если она существует.
        
        Args:
            collection_name (str): Имя коллекции для удаления.
        """
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.drop()
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist.")

    def list_milvus_collections(self):
        """
        Возвращает список всех коллекций в Milvus.
        
        Returns:
            list: Список имен коллекций.
        """
        collections = utility.list_collections()
        logger.info(f"Available collections: {collections}")
        return collections

    def display_first_n_records(self, collection_name, n=5):
        """
        Отображает первые N записей из указанной коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            n (int): Количество записей для отображения.
        """
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return

        collection = Collection(collection_name)
        try:
            results = collection.query(expr=f"id >= 0", output_fields=["id", "text", "chunk_length", "time_insert"], limit=n)
            logger.info(f"First {n} records in collection '{collection_name}':")
            for record in results:
                logger.info(f"ID: {record['id']}, Text: {record['text'][:50]}..., Length: {record['chunk_length']}, Timestamp: {record['time_insert']}")
        except Exception as e:
            logger.error(f"Error fetching records from collection '{collection_name}': {e}")

    def get_total_records(self, collection_name):
        """
        Возвращает общее количество записей в указанной коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
        
        Returns:
            int: Общее количество записей в коллекции.
        """
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return

        collection = Collection(collection_name)
        res = collection.query(
            expr="",
            output_fields=["count(*)"],
        )
        return res[0]["count(*)"]
        
    def insert_data_into_milvus(self, collection_name, chunks, embeddings):
        """
        Вставляет чанки текста и их векторные представления в Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            chunks (list): Список текстовых чанков.
            embeddings (list): Эмбеддинги чанков.
        """
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return

        collection = Collection(collection_name)
        texts = [chunk for chunk in chunks]
        lengths = [len(chunk) for chunk in chunks]
        time_now = int(time.time())
        times = [time_now] * len(chunks)

        data = [
            embeddings,
            texts,
            lengths,
            times
        ]

        try:
            result = collection.insert(data)
            logger.info(f"Inserted {len(result.primary_keys)} records into collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error inserting data into collection '{collection_name}': {e}")

    def search_similar_chunks(self, collection_name, query_embedding):
        """
        Выполняет поиск top_k самых ближайших чанков к заданному запросу в коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            query_embedding (list): Векторный запрос (эмбеддинг).
        
        Returns:
            list: Список кортежей (текст чанка, расстояние до запроса, длина чанка).
        """
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return []

        collection = Collection(collection_name)

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16}
        }

        try:
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=self.top_k,
                output_fields=["text", "chunk_length"]
            )
        except Exception as e:
            logger.error(f"Error during search in collection '{collection_name}': {e}")
            return []

        similar_chunks = []
        seen_text = set()

        for hit in results[0]:
            entity = hit.entity
            distance = hit.distance
            text = entity.get("text")
            chunk_length = entity.get("chunk_length")

            if text not in seen_text:
                seen_text.add(text)
                similar_chunks.append((text, distance, chunk_length))

        return similar_chunks