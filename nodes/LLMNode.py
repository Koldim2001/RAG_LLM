import logging
from elements.QueryElement import QueryElement
from utils_local.utils import profile_time
from langchain_openai import ChatOpenAI  # Используем новый импорт
from langchain.schema import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class LLMNode:
    """Модуль отвечающий, за общение с LLM"""

    def __init__(self, config) -> None:
        self.host = config["host"]
        self.port = config["port"]
        self.model_name = config["model_name"]  
        self.max_tokens = config["max_tokens"]  
        self.temperature = config["temperature"]  
        
        self.openai_api_key = "EMPTY"
        self.openai_api_base = f"http://{self.host}:{self.port}/v1"
    
        # Создаем экземпляр ChatOpenAI из нового пакета
        self.chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
            model_name=self.model_name,  
            max_tokens=self.max_tokens,  
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

        # Получаем ответ от модели
        response = self.chat.invoke(messages)
        query_element.answer = response.content
            
        return query_element


    @profile_time
    def answer(self, query_element: QueryElement, show_data_info=False) -> QueryElement:
        if query_element.previous_messages and query_element.message_number > 0:
            messages = [self.system_prompt] # Системный промпт
            messages.extend(query_element.previous_messages) # История чата
            messages.append(HumanMessage(content=query_element.query)) # Промпт пользователя с примерами
        else:
            messages = [
                self.system_prompt,  # Системный промпт
                HumanMessage(content=query_element.query)  # Промпт пользователя 
            ]
        query_element.final_prompt = messages

        if show_data_info:
            query_element.display_final_prompt()

        # Получаем ответ от модели
        response = self.chat.invoke(messages)
        query_element.answer = response.content
            
        return query_element
    
    @staticmethod
    def get_new_history(query_element: QueryElement) -> list:
        previous_messages = query_element.previous_messages.copy()
        previous_messages.append(HumanMessage(content=query_element.query))
        previous_messages.append(AIMessage(content=query_element.answer))
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