# Scrinium — RAG Chat Architecture

## Overview

Веб-приложение для загрузки документов и RAG-чата с LLM. Часть стэка: docling-mcp (конвертация PDF) + docling-search (ChromaDB) + Scrinium (веб-морда).

## Components

```
User ──→ Browser ──→ FastAPI ──→ ChromaDB
                │           └──→ LLM API (OpenAI-compatible)
                │           └──→ docling (PDF→Markdown)
                │
           scrinium/package
```

### Backend (`scrinium/`)

| Файл | Назначение |
|------|-----------|
| `__main__.py` | Точка входа: FastAPI + uvicorn |
| `__init__.py` | Пакет |
| `convert.py` | Конвертация документов через docling |

### API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/login` | JWT-логин (ADMIN_USER/ADMIN_PASSWORD) |
| POST | `/api/upload` | Загрузить PDF/docx |
| POST | `/api/chat` | Задать вопрос по документам |
| GET | `/api/chat/sessions` | История сессий |
| GET | `/api/documents` | Список загруженных |
| DELETE | `/api/documents/{id}` | Удалить документ |
| DELETE | `/api/chat/sessions/{id}` | Удалить сессию |

### Data Flow

```
1. Upload → docling (OCR/parse) → Markdown → ChromaDB (chunks + embeddings)
2. Chat → embed query → ChromaDB search top-k → LLM prompt → ответ
```

## Dependencies

- **LLM:** Jina embeddings (fastembed) + любой OpenAI-совместимый (OpenRouter, Polza, Ollama)
- **Vector DB:** ChromaDB (встроенная, SQLite-backed)
- **Auth:** bcrypt + PyJWT
- **PDF:** docling (на основе DocTR + Deep Learning, ~2GB RAM)

## Стэк проектов

```
┌──────────┐     ┌───────────┐     ┌──────────┐
│docling-mcp│────▶│ Scrinium  │────▶│  LLM API │
│ (convert) │     │ (web+chat)│     │ (ответы) │
└──────────┘     └──────────┘     └──────────┘
                      │
                      ▼
               ┌──────────────┐
               │docling-search│
               │  (ChromaDB)  │
               └──────────────┘
```

## Configuration (`.env`)

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `HOST` | Адрес сервера | `0.0.0.0` |
| `PORT` | Порт | `9231` |
| `ADMIN_USER` | Логин | `admin` |
| `ADMIN_PASSWORD` | Пароль | — |
| `LLM_BASE_URL` | OpenAI-совместимый API | — |
| `LLM_MODEL` | Модель | — |
| `LLM_API_KEY` | API-ключ | — |
| `CHROMA_DIR` | Путь к ChromaDB | `~/docling-search/chroma` |
