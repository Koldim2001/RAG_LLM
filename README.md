# RAG (Retrieval Augmented Generation) of LLM

В ветке [**GPT_VQA_Example**](https://github.com/Koldim2001/RAG_LLM/tree/GPT_VQA_Example) можно найти примеры инференса готовых моделей.

---
Сделано стримлит приложение, способное осуществлять QA при учитывании контента из нескольких статей из Википедии на темы связанные с компьютерным зрением и Nvidia.

Запуск поисковика с RAG:
```
docker compose up -d --build
```



---
Модели, что использованы в проекте (поднимаются локально):
```
LLM: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
Embedder : intfloat/multilingual-e5-large-instruct
Reranker : BAAI/bge-reranker-v2-m3
```


Как предустановить модель Qwen чтобы вольюм ее увидел и без инета запускал:
```
cd models/nlp/llm
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
```
PS: если заменить `command: --model /models/Qwen2.5-3B-Instruct-GPTQ-Int4` на `--model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` то модель сама скачается в кэш и не надо будет ее через git скачивать с хагингфеса (но тогда при каждом перезапуске будет качать ее)


