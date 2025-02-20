import streamlit as st
from services.MakeDatasetRAG import MakeDatasetRAG

# Инициализация AskLLM через @st.cache_resource
@st.cache_resource
def initialize_dataset():
    return MakeDatasetRAG()

def settings_rag():
    # Получаем объект AskLLM
    make_rag = initialize_dataset()

    st.markdown("# Настройки RAG")

    st.markdown("---")

    # #### Создание коллекций в БД ####
    st.markdown("#### __Создание коллекций в БД__")
    # Большое поле для ввода URL-адресов (можно вводить несколько через Enter)
    url_list_input = st.text_area("Введите URL-адреса (по одному на строку):", height=200)
    # Текстовое поле для ввода имени коллекции
    collection_db_name = st.text_input("Введите имя коллекции для создания RAG:", value="")

    # Проверка, что все поля заполнены
    if st.button("Создать RAG датасет"):
        if not url_list_input.strip() or not collection_db_name.strip():
            st.warning("Пожалуйста, заполните все поля!")
        else:
            try:
                # Преобразуем введенные URL-адреса в список
                url_list = [url.strip() for url in url_list_input.splitlines() if url.strip()]
                
                # Запускаем процесс создания RAG-датасета
                make_rag.process(url_list, collection_db_name)
                st.success(f"RAG датасет '{collection_db_name}' успешно создан.")
            except Exception as e:
                st.error(f"Произошла ошибка при создании RAG датасета: {e}")

    st.markdown("---")

    # #### Удаление коллекций в БД ####
    st.markdown("#### __Удаление коллекций в БД__")
    # Проверяем, есть ли список коллекций в st.session_state
    if "collections" not in st.session_state:
        # Если нет, получаем список коллекций из Milvus и сохраняем его в st.session_state
        st.session_state.collections = make_rag.vector_db_node.list_milvus_collections()
    # Добавляем опцию "Без коллекции" в начало списка
    collections = st.session_state.collections
    # Создание выпадающего меню
    selected_collection = st.selectbox(
        "Выберите коллекцию",
        collections
    )

    if selected_collection != None:
        if st.button("Удалить коллекцию"):
            try:
                # Вызываем метод для удаления коллекции
                make_rag.vector_db_node.delete_milvus_collection(selected_collection)
                st.success(f"Коллекция '{selected_collection}' успешно удалена.")
                
                # Обновляем список коллекций после удаления
                st.session_state.collections = make_rag.vector_db_node.list_milvus_collections()
            except Exception as e:
                st.error(f"Произошла ошибка при удалении коллекции: {e}")