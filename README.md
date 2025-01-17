# RAG (Retrieval Augmented Generation) of LLM

Модели:

LLM: Qwen2.5-3B-Instruct-GPTQ-Int4


Переменные окружнения:
'''
MODEL_PATH_EMB=intfloat/multilingual-e5-large-instruct
MODEL_PATH_RETRIEVAL=BAAI/bge-reranker-v2-m3
```

Запуск:
```
docker compose up -d --build
```