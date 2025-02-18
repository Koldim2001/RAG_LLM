import logging
import time
from pymilvus import FieldSchema, CollectionSchema, DataType
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class VectorDBNode:
    """Модуль отвечающий, за работу с веторной базой данных"""

    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.dim = config["dim"]
        # Создание клиента Milvus
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        logger.info(f"Connected to Milvus at {self.host}:{self.port} successfully!")

    def create_milvus_collection(self, collection_name):
        """
        Создает новую коллекцию в Milvus, если она еще не существует.
        
        Args:
            collection_name (str): Имя коллекции.
            dim (int): Размерность векторов.
        """
        # Проверяем, существует ли коллекция
        if self.client.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return
        
        # Определение полей коллекции
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_length", dtype=DataType.INT64),
            FieldSchema(name="time_insert", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="Collection for text chunks")
        
        # Создание коллекции
        self.client.create_collection(collection_name, schema)
        logger.info(f"Collection '{collection_name}' created successfully.")

    def create_index(self, collection_name, field_name="embedding", index_params=None):
        """
        Создает индекс на указанном поле в коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            field_name (str): Поле, на котором создается индекс (по умолчанию "embedding").
            index_params (dict): Параметры индекса (по умолчанию IVF_FLAT).
        """
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return
        
        # Устанавливаем параметры индекса по умолчанию (IVF_FLAT)
        if index_params is None:
            index_params = {
                "metric_type": "IP",  # Метрика внутреннего произведения
                "index_type": "IVF_FLAT",  # Тип индекса
                "params": {"nlist": 128}  # Количество кластеров
            }

        # Создаем индекс
        self.client.create_index(collection_name, field_name, index_params)
        logger.info(f"Index created on field '{field_name}' for collection '{collection_name}'.")

        # Загружаем коллекцию для применения индекса
        self.client.load_collection(collection_name)
        logger.info(f"Collection '{collection_name}' loaded successfully.")
        
        self.create_index(collection_name)

    def delete_milvus_collection(self, collection_name):
        """
        Удаляет коллекцию из Milvus, если она существует.
        
        Args:
            collection_name (str): Имя коллекции для удаления.
        """
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist.")

    def list_milvus_collections(self):
        """
        Возвращает список всех коллекций в Milvus.
        
        Returns:
            list: Список имен коллекций.
        """
        collections = self.client.list_collections()
        logger.info(f"Available collections: {collections}")
        return collections
    
    def display_first_n_records(self, collection_name, n=5):
        """
        Отображает первые N записей из указанной коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            n (int): Количество записей для отображения.
        """
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return
        
        # Выполняем запрос для получения первых N записей
        try:
            results = self.client.query(
                collection_name=collection_name,
                expr=f"id >= 0",  # Базовое условие для выборки всех записей
                output_fields=["id", "text", "chunk_length", "timestamp"],
                limit=n
            )
            logger.info(f"First {n} records in collection '{collection_name}':")
            for record in results:
                logger.info(f"ID: {record['id']}, Text: {record['text'][:50]}..., Length: {record['chunk_length']}, Timestamp: {record['timestamp']}")
        except Exception as e:
            logger.error(f"Error fetching records from collection '{collection_name}': {e}")

    def insert_data_into_milvus(self, collection_name, chunks, embeddings):
        """
        Вставляет чанки текста и их векторные представления в Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            chunks (list): Список текстовых чанков.
            embeddings (list): эмбеддинги чанков
        """
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return
        
        # Подготовка данных для вставки
        texts = [chunk for chunk in chunks]
        lengths = [len(chunk) for chunk in chunks]
        time_now = int(time.time())
        times = [time_now] * len(chunks)

        data = [
            embeddings,  # Векторы
            texts,       # Тексты чанков
            lengths,     # Длины чанков
            times        # Временные метки
        ]

        # Вставка данных в коллекцию
        result = self.client.insert(collection_name, data)
        logger.info(f"Inserted {len(result.primary_keys)} records into collection '{collection_name}'.")

   
    def search_similar_chunks(self, collection_name, query_embedding, top_k=15):
        """
        Выполняет поиск top_k самых ближайших чанков к заданному запросу в коллекции Milvus.
        
        Args:
            collection_name (str): Имя коллекции.
            query_embedding (list): Векторный запрос (эмбеддинг).
            top_k (int): Количество ближайших чанков для поиска.
        
        Returns:
            list: Список кортежей (текст чанка, расстояние до запроса, длина чанка).
        """
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            return []

        # Проверяем, создан ли индекс на поле "embedding"
        indexes = self.client.describe_index(collection_name)
        if not indexes or "embedding" not in [index["field_name"] for index in indexes]:
            logger.warning(f"No index found on 'embedding' field. Creating an index...")
            self.create_index(collection_name, field_name="embedding")

        # Загружаем коллекцию, если она еще не загружена
        if not self.client.is_collection_loaded(collection_name):
            self.client.load_collection(collection_name)

        # Параметры поиска
        search_params = {
            "metric_type": "IP",  # Метрика внутреннего произведения (для косинусной близости)
            "params": {"nprobe": 16}  # Параметр для оптимизации поиска
        }

        # Выполняем поиск
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],  # Список запросов (векторов)
                anns_field="embedding",  # Поле для поиска (векторное представление)
                param=search_params,
                limit=top_k,  # Количество результатов
                output_fields=["text", "chunk_length"]  # Дополнительные поля для вывода
            )
        except Exception as e:
            logger.error(f"Error during search in collection '{collection_name}': {e}")
            return []

        # Обработка результатов
        similar_chunks = []
        seen_text = set()  # Множество для отслеживания уникальных текстов
        for result in results: 
            entity = result.entity
            distance = result.distance
            text = entity.get("text")
            chunk_length = entity.get("chunk_length")

            # Проверяем, не встречался ли этот текст ранее
            if text not in seen_text:
                seen_text.add(text)  # Добавляем текст в множество
                similar_chunks.append((text, distance, chunk_length))

        return similar_chunks