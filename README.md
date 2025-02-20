# RAG (Retrieval Augmented Generation) with LLM
---
Сделано приложение, способное осуществлять QA при учитывании контента из любых предоставляемых сайтов. Надо передать список ссылок и далее можно задавать вопросы по данному материалу.

Запуск сервисов:
```
docker compose up -d --build
```
![Без имени](https://github.com/user-attachments/assets/8d8da6d9-8c11-4dd3-b611-be1675e4ea74)
Сайт по работе с LLM (чат-бот с рагом) станет доступен после запуска компоуза по этому адресу - http://localhost:8501/

Код для работы с раг из python - *rag_example.ipynb*

Туториал по проекту - [видео](https://rutube.ru/video/private/236899f9912c7ebaabd3f4142c672684/?p=00pHeMu2UAZse16c78_wDA)

---
Модели, что использованы в проекте (поднимаются локально):
```
LLM: Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
Embedder : intfloat/multilingual-e5-large-instruct
Reranker : BAAI/bge-reranker-v2-m3
```

Как предустановить модель Qwen чтобы вольюм ее увидел и без инета запускал:
```
cd models/nlp/llm
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
```
PS: если заменить `command: --model /models/Qwen2.5-7B-Instruct-GPTQ-Int4` на `--model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` то модель сама скачается в кэш и не надо будет ее через git скачивать с хагингфеса (но тогда при каждом перезапуске будет качать ее)

---

В ветке [**GPT_VQA_Example**](https://github.com/Koldim2001/RAG_LLM/tree/GPT_VQA_Example) можно найти примеры инференса готовых моделей в python по api. В том числе там пример с Visual LLM (VQA).


