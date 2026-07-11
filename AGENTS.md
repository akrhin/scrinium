# Scrinium — Agent Instructions

## Что это

Веб-морда для RAG-чата по документам. FastAPI + ChromaDB + docling.
Загрузка PDF → конвертация в Markdown → эмбеддинги в ChromaDB → чат с LLM.

## Быстрый старт

```bash
cd ~/git/scrinium
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # заполнить LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
python3 -m scrinium   # → http://localhost:9231
```

## Архитектура

См. `ARCHITECTURE.md`

## Структура

```
scrinium/
├── scrinium/
│   ├── __main__.py    # FastAPI + uvicorn
│   ├── __init__.py    # пакет
│   └── convert.py     # docling-конвертация
├── data/              # загруженные документы
├── .env               # конфиг
├── pyproject.toml
└── requirements.txt
```

## CI/CD

Нет CI — локальный проект.

## Зависимости

- **ChromaDB + fastembed + jina-embeddings-v3** — ~700 MB RAM
- **docling** — ~2 GB RAM (только при загрузке PDF)
- **LLM API** — любой OpenAI-совместимый

## Port

`9231` — выбран для отличия от docling-mcp и других сервисов.

## Известные проблемы

1. **ChromaDB shared с docling-search** — если оба запущены, ChromaDB может блокироваться. Решение: использовать `CHROMA_DIR` явно.
2. **docling медленный** — при загрузке большого PDF (сотни страниц) конвертация может занимать минуты.
3. **fastembed загружает модель в RAM** — при первом запуске скачивает jina-embeddings-v3 (~700MB), без интернета не работает.
