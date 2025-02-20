import streamlit as st
from services.AskLLM import AskLLM

# Инициализация AskLLM через @st.cache_resource
@st.cache_resource
def initialize_ask_llm():
    return AskLLM()

# Получаем объект AskLLM
ask = initialize_ask_llm()

# Инициализация состояния сессии для хранения истории чата Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Левая панель (sidebar) для выбора collection_db_name
with st.sidebar:
    # Проверяем, есть ли список коллекций в st.session_state
    if "collections" not in st.session_state:
        # Если нет, получаем список коллекций из Milvus и сохраняем его в st.session_state
        st.session_state.collections = ask.vector_db_node.list_milvus_collections()
    
    # Добавляем опцию "Без коллекции" в начало списка
    collections = ["Без коллекции"] + st.session_state.collections
    
    # Создание выпадающего меню
    selected_collection = st.selectbox(
        "Выберите RAG коллекцию",
        collections,
        index=0  # По умолчанию выбирается "Без коллекции"
    )
    
    # Установка collection_db_name
    if selected_collection == "Без коллекции":
        collection_db_name = None
        st.markdown("__Mодель будет отвечать без RAG__")
    else:
        collection_db_name = selected_collection


# Отображение истории чата
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Принятие пользовательского ввода
if prompt := st.chat_input("Задайте ваш вопрос:"):
    # Добавление сообщения пользователя в историю чата Streamlit
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Подготовка previous_messages для AskLLM
    if "previous_messages" not in st.session_state:
        st.session_state.previous_messages = []  # Инициализация внутренней истории для AskLLM

    # Обработка запроса через AskLLM
    message_number = (len(st.session_state.chat_history) - 1) // 2  # Номер текущего сообщения
    res = ask.process(prompt, message_number, collection_db_name, st.session_state.previous_messages)

    # Получение ответа модели
    answer = res.answer
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Обновление внутренней истории для AskLLM
    st.session_state.previous_messages = ask.get_new_history(res)

# Очистка истории при обновлении страницы
if st.sidebar.button("Очистить историю чата"):
    st.session_state.chat_history = []
    st.session_state.previous_messages = []