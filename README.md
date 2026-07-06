# Scrinium — RAG Chat over your documents

Веб-интерфейс для загрузки документов и RAG-чата с LLM.

**Часть стэка:**

| Компонент | Назначение | Как запускается |
|-----------|-----------|-----------------|
| **docling-mcp** | Конвертер PDF → Markdown | MCP-сервер (через Hermes) |
| **docling-search** | Семантический поиск (ChromaDB) | MCP-сервер (через Hermes) |
| **Scrinium** | Веб-морда + чат с LLM | `python3 -m scrinium` |

**Общие данные:**
- ChromaDB: `~/docling-search/chroma/`
- Модель эмбеддингов: `jinaai/jina-embeddings-v3` (fastembed)
- Что загрузил через веб — доступно через MCP-инструменты, и наоборот

## Быстрый старт

```bash
# 1. Клонировать
git clone https://github.com/akrhin/scrinium.git
cd scrinium

# 2. Виртуальное окружение
uv venv
source .venv/bin/activate

# 3. Зависимости
uv pip install -r requirements.txt

# 4. Настройки
cp .env.example .env
# Заполни .env: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, ADMIN_PASSWORD

# 5. Запуск
python3 -m scrinium
# → http://localhost:9231
```

## Настройка

```bash
# .env
ADMIN_USER=sintez
ADMIN_PASSWORD=мой_пароль

LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/free
LLM_API_KEY=sk-***

# ChromaDB уже общая с docling-search
# Можно не трогать, если docling-search тоже на этом хосте
# CHROMA_DIR=/home/sintez/docling-search/chroma
```

## API

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/login` | JWT-логин |
| POST | `/api/upload` | Загрузить PDF/docx/… |
| POST | `/api/chat` | Задать вопрос |
| GET | `/api/chat/sessions` | История сессий |
| GET | `/api/documents` | Список документов |
| DELETE | `/api/documents/{id}` | Удалить документ |
| DELETE | `/api/chat/sessions/{id}` | Удалить сессию |

## Требования

- Python ≥ 3.10
- ChromaDB (ставится автоматически)
- fastembed + jina-embeddings-v3 (ставится автоматически, ~700 MB RAM)
- Для PDF: docling (ставится автоматически, ~2 GB RAM)
- Для AI-ответов: OpenAI-совместимый LLM API (OpenRouter, Polza, локальный Ollama)

## Лицензия

GNU General Public License v3.0
