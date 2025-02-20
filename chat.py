import streamlit as st
from services.AskLLM import AskLLM

# Инициализация AskLLM
ask = AskLLM()

# Инициализация состояния сессии для хранения истории чата Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Левая панель (sidebar) для выбора collection_db_name
with st.sidebar:
    collection_db_name = st.text_input("RAG коллекция", value="", placeholder="Введите имя коллекции в БД")
    if not collection_db_name:
        collection_db_name = None  # Если поле пустое, устанавливаем значение None

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
    message_number = (len(st.session_state.chat_history) - 1) // 2 # Номер текущего сообщения
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