
class QueryElement:
    # Класс, содержаций информацию о конкретном запросе
    def __init__(
        self,
        query: str,
        message_number: int = 0,
        collection_db_name: str | None = None,
        previous_messages: list = [],
        query_embedding: list | None = None,
        top_chunks: list | None = None,
        prompt_chunks: list | None = None,
        final_prompt: str | None = None,
        answer: str | None = None,
    ) -> None:
        self.query = query # Запрос пользователя
        self.message_number = message_number # Номер сообщения в чате (начиная с 0)
        self.collection_db_name = collection_db_name  # Имя коллекции для векторной БД для хранения данных
        self.previous_messages = previous_messages  # История чата
        self.upgraded_query = query # Запрос пользователя с добавленным контекстом чата (по умолчанию исходный запрос)
        self.query_embedding = query_embedding # эмбеддинг от upgraded_query
        self.top_chunks = top_chunks # проранжированные чанки (топ лучших)
        self.prompt_chunks = prompt_chunks # итоговые чанки, добавляемые в запрос (прошедшие фильтрацию по релевантности и числу)
        self.final_prompt = final_prompt # итоговый промпт
        self.answer = answer # ответ llm
  
    @staticmethod
    def display_chunks(chunks: list, limit_size: int = 200) -> None:
        """
        Выводит информацию о чанках с ограничением длины текста.
        
        :param chunks: Список чанков для вывода.
        :param limit_size: Максимальное количество символов для отображения каждого чанка (по умолчанию 200).
        """
        if not chunks:
            print("Нет доступных чанков для отображения.")
            return

        for i, chunk in enumerate(chunks):
            content = chunk.page_content[:limit_size] + "..." if len(chunk.page_content) > limit_size else chunk.page_content
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            print(f"Chunk {i + 1}: {content}, \nmetadata={metadata}\n")


    def display_final_prompt(self) -> None:
        """
        Выводит итоговый промпт, подаваемый на вход модели.
        """
        if not self.final_prompt:
            print("Итоговый промпт отсутствует.")
            return

        if isinstance(self.final_prompt, list):
            # каждый элемент списка — это словарь с ключами 'type' и 'content'
            print("Итоговый промпт, подаваемый на вход модели:\n")
            for message in self.final_prompt:
                if 'type' in message and 'content' in message:
                    print(f"{message['type']}: {message['content']}\n")
                else:
                    print(f"Неверный формат сообщения: {message}")
        else:
            print(f"Неподдерживаемый формат итогового промпта: {type(self.final_prompt)}")