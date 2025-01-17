# RAG (Retrieval Augmented Generation) of LLM

Модели:
```
LLM: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
Embedder : intfloat/multilingual-e5-large-instruct
Reranker : BAAI/bge-reranker-v2-m3
```

Запуск:
```
docker compose up -d --build
```