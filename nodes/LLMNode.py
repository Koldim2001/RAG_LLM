import logging
import requests
from elements.QueryElement import QueryElement
from utils_local.utils import profile_time
from langchain_openai import ChatOpenAI  
from langchain.schema import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class LLMNode:
    """Модуль отвечающий, за общение с LLM"""

    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.model_name = config["model_name"]  
        self.max_tokens_output = config["max_tokens_output"] 
        self.max_tokens_input = config["max_tokens_input"] 
        self.temperature = config["temperature"]  
        self.max_messages_history = config["max_messages_history"]  
        
        self.openai_api_key = "EMPTY"
        self.openai_api_base = f"http://{self.host}:{self.port}/v1"
    
        # Создаем экземпляр ChatOpenAI из нового пакета
        self.chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            model_name=self.model_name,  
            max_tokens=self.max_tokens_output,  
            temperature=self.temperature  
        )

        # Системный промпт, который задает контекст для модели
        self.system_prompt = SystemMessage(content="Вы полезный помощник, который отвечает на вопросы на русском языке. Ваши ответы должны быть четкими, информативными и полезными.")


    @profile_time
    def answer_with_rag(self, query_element: QueryElement, show_data_info=False) -> QueryElement:
        prompt_chunks = query_element.prompt_chunks
        
        examples = "\n".join([f"{i}. {doc.page_content}" for i, doc in enumerate(prompt_chunks, start=1)])

        if len(prompt_chunks) > 0:
            # Формируем итоговый промпт
            human_message_content = (
                f"Учитывай информацию из этих отрывков текста если считаешь нужным:\n"
                f"{examples}\n\n"
                f"Ответь на вопрос: {query_element.query}"
            )
        else:
            # Формируем итоговый промпт
            human_message_content = (
                f"Ответь на вопрос: {query_element.query}\n"
                f"Обязательно укажи, что отвечаешь без учета контекста с представленных сайтов, так как не нашел ничего релевантного"
            )

        if query_element.previous_messages and query_element.message_number > 0:
            messages = [self.system_prompt] # Системный промпт
            messages.extend(query_element.previous_messages) # История чата
            messages.append(HumanMessage(content=human_message_content)) # Промпт пользователя с примерами
        else:
            messages = [
                self.system_prompt,  # Системный промпт
                HumanMessage(content=human_message_content)  # Промпт пользователя с примерами
            ]

        query_element.final_prompt = messages

        if show_data_info:
            query_element.display_final_prompt()

        # Проверяем размер запроса
        try:
            self._validate_prompt_size(messages)
        except ValueError as e:
            return query_element  # Возвращаем объект без ответа

        # Получаем ответ от модели
        response = self.chat.invoke(messages)
        query_element.answer = response.content
            
        return query_element

    @profile_time
    def answer(self, query_element: QueryElement, show_data_info=False) -> QueryElement:
        if query_element.previous_messages and query_element.message_number > 0:
            messages = [self.system_prompt] # Системный промпт
            messages.extend(query_element.previous_messages) # История чата
            messages.append(HumanMessage(content=query_element.query)) # Промпт пользователя
        else:
            messages = [
                self.system_prompt,  # Системный промпт
                HumanMessage(content=query_element.query)  # Промпт пользователя 
            ]
        query_element.final_prompt = messages

        if show_data_info:
            query_element.display_final_prompt()

        # Проверяем размер запроса
        try:
            self._validate_prompt_size(messages)
        except ValueError as e:
            return query_element  # Возвращаем объект с ошибкой

        # Получаем ответ от модели
        response = self.chat.invoke(messages)
        query_element.answer = response.content
            
        return query_element
    
    def get_new_history(self, query_element: QueryElement) -> list:
        previous_messages = query_element.previous_messages.copy()
        previous_messages.append(HumanMessage(content=query_element.query))
        previous_messages.append(AIMessage(content=query_element.answer))

        max_len = 2 * self.max_messages_history
        # Ограничиваем количество сообщений до max_len, если история слишком большая
        if len(previous_messages) > max_len:
            previous_messages = previous_messages[-max_len:]
    
        return previous_messages
    
    @profile_time
    def make_abstract(self, query_element: QueryElement) -> str:
        if query_element.previous_messages and query_element.message_number > 0:
            messages = [
                SystemMessage(content="Вы полезный помощник, который отвечает на вопросы очень кратко в одно предложение.")
            ]
            messages.extend(query_element.previous_messages)  # История чата
            # Уточненный промпт
            prompt = (
                f'Перефразируй запрос пользователя, дополнив его информацией из контекста прошлых сообщений. '
                f'Не отвечай на сам запрос, только уточни или дополни его! '
                f'Запрос пользователя: "{query_element.query}"'
            )
            messages.append(HumanMessage(content=prompt))  # Промпт пользователя
            
            # Получаем ответ от модели
            response = self.chat.invoke(messages)
            abstract = response.content[:1000]
            logger.info(f"Измененный текст сообщения для получения эмеддингов: {abstract}")
            return abstract
        else:
            return query_element.query

    def count_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в заданном тексте через эндпоинт /tokenize.

        :param text: Текст, для которого нужно подсчитать токены.
        :return: Количество токенов в тексте.
        """
        url = f"http://{self.host}:{self.port}/tokenize"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Проверяем успешность запроса
            result = response.json()
            token_count = len(result.get("tokens", []))  # Получаем количество токенов
            return token_count
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при подсчете токенов: {e}")
            return -1  # Возвращаем -1 в случае ошибки
        
    def _messages_to_text(self, messages: list) -> str:
        """
        Преобразует список сообщений в одну строку для подсчета символов и токенов.
        :param messages: Список сообщений.
        :return: Объединенный текст всех сообщений.
        """
        text_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                text_parts.append(f"<|system|>{message.content}<|end|>")
            elif isinstance(message, HumanMessage):
                text_parts.append(f"<|user|>{message.content}<|end|>")
            elif isinstance(message, AIMessage):
                text_parts.append(f"<|assistant|>{message.content}<|end|>")
        return "\n".join(text_parts)
    
    def _validate_prompt_size(self, messages: list):
        """
        Проверяет размер запроса (символы и токены) и выбрасывает ошибку, если он превышает лимит.
        :param messages: Список сообщений для проверки.
        """
        # Преобразуем сообщения в строку для подсчета символов и токенов
        full_prompt_text = self._messages_to_text(messages)

        # Подсчет длины в символах
        symbols_count = len(full_prompt_text)
        tokens_count = self.count_tokens(full_prompt_text)
        logging.info(f"Размер запроса: {symbols_count} символов, {tokens_count} токенов")

        # Проверяем лимит по токенам
        if tokens_count > self.max_tokens_input:
            logging.warning(f"Превышение лимита токенов на запрос: {tokens_count} > {self.max_tokens_input}")
            raise ValueError(f"Размер запроса слишком большой: {symbols_count} символов, {tokens_count} токенов. Максимальный лимит: {self.max_tokens_input} токенов.")
