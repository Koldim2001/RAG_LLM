import streamlit as st
from streamlit_pages.chat import chat
from streamlit_pages.settings_rag import settings_rag
from streamlit_option_menu import option_menu


def main():
    st.set_page_config(
    page_title="Chat-bot-RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",)

    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    with st.sidebar:        
        app = option_menu(
            menu_title='Чат-бот c RAG',
            options=['Чат','Настройка RAG'],
            icons=['bi bi-1-square-fill','bi bi-2-square-fill'],
            default_index=0,
            styles={"container": {"padding": "5!important","background-color":'gray'},
            "icon": {"color": "white", "font-size": "15px"}, 
            "nav-link": {"color":"white","font-size": "13px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
            "nav-link-selected": {"background-color": "#02ab21"},
            "menu-title": {"color": "white", "font-size": "16px"},
            "menu-title-icon": {"display": "none"} }   
            )
    if app == 'Чат':
        chat()
    if app == 'Настройка RAG':
        settings_rag()


if __name__ == "__main__":
    main()
