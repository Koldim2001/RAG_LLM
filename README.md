# Примеры инференса готовых моделей

### Модели:
```
LLM: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
VQA: Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4 
Embedder : intfloat/multilingual-e5-large-instruct
Reranker : BAAI/bge-reranker-v2-m3
```
#### код с примерами - *test_gpt.ipynb*

Как предустановить модель Qwen чтобы вольюм ее увидел и без инета запускал (ps: VQA модель качать идентично только в папку models/nlp/vqa):
```
cd models/nlp/llm
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
```
PS: если заменить `command: --model /models/Qwen2.5-3B-Instruct-GPTQ-Int4` на `--model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` то модель сама скачается в кэш и не надо будет ее через git скачивать с хагингфеса (но тогда при каждом перезапуске будет качать ее)

Запуск:
```
docker compose up -d --build
```

