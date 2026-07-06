# Hermes RAG Stack — архитектура

> Документация стека: Hermes Agent + Scrinium + docling-mcp + docling-search + ChromaDB
>
> Сервер: Debian 13, Intel Xeon E3-1265L v3 (4 ядра), 8 GB RAM, 96 GB SSD

---

## 1. Общая схема

```
Wi-Fi / LAN
  │
  ▼
┌──────────────── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│  Ты (V) через браузер или Telegram                 │
│  ├── http://192.168.1.92:9231 — Scrinium (веб)    │
│  └── Telegram — я (Hermes Agent)                   │
└──────────────────────┬─────────────────────────────┘
                       │
═══════════════════════╪═══════════════════ Сервер ═══
                       │
┌──────────────────────▼──────────────────────────────┐
│  Hermes Agent (:8642)                               │
│  ─────────────────────                              │
│  Моя основная среда. Запущен через systemd.          │
│  Управляет MCP-серверами и cron-задачами.            │
│                                                      │
│  MCP-серверы (stdio):                                │
│  ├── docling-mcp     — конвертация PDF → Markdown   │
│  └── docling-search  — семантический поиск           │
│                                                      │
│  Провайдер: polza (DeepSeek V4 Flash)                │
│  Память: Mnemosyne (embed: text-embedding-3-small)   │
└──────────────────────┬──────────────────────────────┘
                       │
     ┌─────────────────┼──────────────────┐
     ▼                 ▼                   ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────┐
│Scrinium  │  │ Docling      │  │ Polz-proxy        │
│:9231     │  │ (subprocess) │  │ :8787             │
│FastAPI   │  │ convert.py   │  │ API-прокси        │
│веб-морда │  │ PDF→Markdown │  │ (LLM + эмбеддинги)│
└────┬─────┘  └──────┬───────┘  └────────┬─────────┘
     │               │                    │
     ▼               ▼                    ▼
┌─────────────────────────────────────────────────────┐
│  ChromaDB (~/docling-search/chroma/)                  │
│  ────────────────────────                             │
│  Одна коллекция:  docs                                │
│  Модель:          text-embedding-3-small / 1024 dim   │
│  Provider:        remote (Polz-proxy HTTP API)        │
│  Схема:           метаданные {doc_id, doc_name,      │
│                   header, seq, source}                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. Компоненты

### 2.1. Hermes Agent

| Параметр | Значение |
|----------|----------|
| Версия | v0.18.0+ |
| Провайдер | polza (DeepSeek V4 Flash) |
| Порт API | 8642 |
| Конфиг | `~/.hermes/config.yaml` |
| .env | `~/.hermes/.env` |
| Профиль | default |
| MCP-серверы | docling-mcp, docling-search (stdio) |
| Память | Mnemosyne (локальная) |
| Каналы | Telegram, Home Assistant, локальный API |

**Запуск:** systemd (`hermes.service`), автостарт при загрузке.

**Mnemosyne использует ту же модель эмбеддингов:**
```
MNEMOSYNE_EMBEDDING_MODEL=openai/text-embedding-3-small
MNEMOSYNE_EMBEDDING_API_URL=http://127.0.0.1:8787
MNEMOSYNE_EMBEDDING_API_KEY=pza_...
MNEMOSYNE_EMBEDDINGS_VIA_API=true
MNEMOSYNE_EMBEDDING_DIM=1536
```

---

### 2.2. Scrinium

| Параметр | Значение |
|----------|----------|
| Репозиторий | `https://github.com/akrhin/scrinium` (публичный) |
| Путь | `/home/sintez/git/scrinium/` |
| Веб-порт | 9231 |
| Базовый URL | http://192.168.1.92:9231 |
| Python | 3.11 (venv) |
| Веб-фреймворк | FastAPI + Uvicorn |
| База данных | SQLite (`data/scrinium.db`) |
| Фронтенд | HTML + Vanilla JS (встроенный шаблон) |
| CORS | `allow_origins=["*"]` |
| Запуск | `python3 -m scrinium` (foreground) |
| .env | `/home/sintez/git/scrinium/.env` (в .gitignore) |

**Конфигурация .env:**
```ini
ADMIN_USER=sintez
ADMIN_PASSWORD=kWdeZZhbh7SbN3hd63WD3vezy6pPZKK1
JWT_SECRET=scrinium-dev-secret-change-in-prod

# Embedding: внешний API (0 RAM на сервере)
EMBEDDING_PROVIDER=remote
EMBEDDING_API_URL=http://127.0.0.1:8787
EMBEDDING_API_KEY=pza_...
EMBEDDING_API_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# LLM
LLM_BASE_URL=http://127.0.0.1:8787
LLM_MODEL=openrouter/free
LLM_API_KEY=pza_...

# Docling: local (subprocess convert.py)
DOCLING_PROVIDER=local

# Хранилища
CHROMA_DIR=/home/sintez/docling-search/chroma
DB_PATH=./data/scrinium.db
UPLOAD_DIR=./data/uploads
```

**Зависимости (requirements.txt):**
```txt
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.12
pyjwt>=2.9
bcrypt>=4.2
chromadb>=0.5
sqlalchemy>=2.0
httpx>=0.27
python-dotenv>=1.1
aiofiles>=24.1
docling>=2.0
fastembed>=0.8
numpy>=2.0
```

**Структура:**
```
scrinium/
├── scrinium/
│   ├── __init__.py      # FastAPI-приложение + вся логика
│   ├── convert.py       # docling-конвертер (подпроцесс)
│   └── templates/
│       └── index.html   # Фронтенд (самодостаточный SPA)
├── data/
│   ├── scrinium.db      # SQLite с сессиями чата
│   └── uploads/         # Загруженные PDF до конвертации
├── .env                 # Конфигурация (в .gitignore)
├── .env.example         # Шаблон конфига
└── requirements.txt
```

**Запуск:**
```bash
cd /home/sintez/git/scrinium
source .venv/bin/activate
python3 -m scrinium    # → :9231
```

**API-эндпоинты:**

| Метод | Путь | Назначение |
|-------|------|-----------|
| GET | `/` | Главная страница |
| POST | `/api/login` | Аутентификация |
| GET | `/api/me` | Проверка токена |
| POST | `/api/upload` | Загрузка файла (multipart) |
| GET | `/api/documents` | Список документов |
| DELETE | `/api/documents/{id}` | Удаление документа |
| POST | `/api/chat` | Отправить сообщение в чат |
| GET | `/api/chat/sessions` | Список чатов |
| GET | `/api/chat/sessions/{id}` | История чата |
| DELETE | `/api/chat/sessions/{id}` | Удалить чат |
| GET | `/api/llm/settings` | Настройки LLM |

**Поток загрузки:**
```
1. Браузер → multipart POST /api/upload (через XHR с прогресс-баром)
2. Scrinium сохраняет файл в data/uploads/
3. Если text/md/html → читает напрямую
   Если PDF/DOCX/EPUB → запускает convert.py (subprocess)
4. Текст → чанкинг (200-2000 символов, по параграфам)
5. Чанки → _embed_texts() → Polz-proxy → векторы 1024d
6. Векторы → ChromaDB (коллекция "docs")
7. Ответ: {doc_id, chunks, pages, filename}
```

---

### 2.3. Docling-MCP (конвертер)

| Параметр | Значение |
|----------|----------|
| Проект | https://github.com/docling-project/docling-mcp |
| Репозиторий | `https://github.com/akrhin/docling-mcp` (приватный) |
| Путь | `/home/sintez/git/docling-mcp/` |
| Python | 3.13 (venv) |
| Режим | local (CPU, ONNX) |
| Модели | Docling + RapidOCR (сканы) |
| Подпроцесс | нет — живёт как stdio MCP-сервер |
| RAM | ~2 GB при конвертации (освобождается после) |
| Таймаут | 600 секунд |

**Запуск:** автоматически, через Hermes Gateway.
**Обёртка:** `mcp_wrapper.sh` — чистит CWD и ставит `DOCLING_CONVERSION_MODE=local`.

**Что конвертит:** PDF (текст + сканы), DOCX, PPTX, XLSX, HTML, изображения, URL.

**MCP-инструменты (для меня):**
| Инструмент | Назначение |
|-----------|-----------|
| `convert_document_into_docling_document` | Конвертировать файл/URL → Markdown |
| `is_document_in_local_cache` | Проверить кеш |
| `convert_directory_files_into_docling_document` | Пакетная конвертация |

---

### 2.4. Docling-Search (поиск)

| Параметр | Значение |
|----------|----------|
| Путь | `/home/sintez/git/docling-mcp/docling_search.py` |
| Python | 3.13 (venv) |
| Язык | Python + asyncio |
| База | ChromaDB (persistent) |
| Путь ChromaDB | `~/docling-search/chroma/` |
| Коллекция | `docs` |
| Размерность | 1024 |
| Эмбеддинги | через Polz-proxy (HTTP API) |
| Режим | async + batch |

**Env-переменные (через Hermes config.addon.yaml):**
```yaml
EMBED_API_URL: http://127.0.0.1:8787
EMBED_API_KEY: pza_...
EMBED_MODEL: text-embedding-3-small
```

**MCP-инструменты (для меня):**

| Инструмент | Назначение |
|-----------|-----------|
| `index_doc(path)` | Индексировать .md файл |
| `search_docs(query)` | Семантический поиск |
| `list_docs` | Список документов |
| `get_doc(doc_id)` | Полный текст |
| `get_chunk(chunk_id)` | Конкретный раздел |
| `delete_doc(doc_id)` | Удалить из индекса |

**Как я ищу:**
```
1. User: "найди в документах про VPN"
2. Я → mcp__docling_search__search_docs(query="VPN")
3. docling_search.py → POST /embeddings → Polz-proxy → вектор 1024d
4. → ChromaDB query → топ-5 чанков
5. → возвращает мне тексты + метаданные
6. Я формирую ответ с цитатами
```

---

### 2.5. Polz-proxy (API-прокси)

| Параметр | Значение |
|----------|----------|
| Адрес | http://127.0.0.1:8787 |
| Порт | 8787 |
| Назначение | OpenAI-совместимый прокси для LLM и эмбеддингов |
| Ключ | `pza_...` |
| Статус | запущен (systemd или ручной старт) |
| Эндпоинты | `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |

**Модели эмбеддингов (доступные через прокси):**
- `text-embedding-3-small` (1536 → 1024 dim) ← **используется**
- `text-embedding-3-large` (3072 → 1024 dim)
- `text-embedding-ada-002`
- `qwen/qwen3-embedding-4b`
- `google/gemini-embedding-2-preview`
- `mistralai/mistral-embed-2312`
- И ещё 5 моделей

**Кто использует Polz-proxy для эмбеддингов:**
| Компонент | Модель | Размерность |
|-----------|--------|-------------|
| Mnemosyne | text-embedding-3-small | 1536 |
| Scrinium | text-embedding-3-small | 1024 |
| docling-search | text-embedding-3-small | 1024 |

> ⚠️ **Важно:** размерность должна совпадать с коллекцией в ChromaDB.
> Scrinium и docling-search используют 1024 dim; Mnemosyne — 1536.
> При переключении модели — очистка ChromaDB обязательна.

---

### 2.6. ChromaDB (векторная БД)

| Параметр | Значение |
|----------|----------|
| Тип | PersistentClient (файловая) |
| Путь | `~/docling-search/chroma/` |
| Коллекция | `docs` |
| Метрика | cosine similarity |
| Размерность | 1024 |
| Batch size | 100 (добавление), 20 (эмбеддинг) |
| Shared между | Scrinium + docling-search |
| Data | Markdown-чанки + метаданные |

**Схема метаданных:**
```json
{
  "doc_id": "a1b2c3d4",       // UUID первой 8 символов
  "doc_name": "manual-ru",     // Имя документа
  "header": "Глава 3",         // Заголовок секции
  "seq": 7,                    // Номер чанка
  "source": "document.pdf"     // Исходный файл
}
```

**Как работает разделение:**
- Scrinium пишет в `docs` через HTTP API (свои вызовы ChromaDB)
- docling-search пишет в `docs` через прямой вызов ChromaDB
- Я ищу через docling-search (MCP) — он читает из `docs`
- Итог: **единая база**, кто бы ни загрузил документ

---

## 3. Связи компонентов

```
┌─────────────────────────────────────────────────────┐
│  Hermes Agent                                       │
│                                                      │
│  ┌────────────┐  ┌──────────────────────────────┐   │
│  │ Mnemosyne  │  │ MCP Gateway                  │   │
│  │ (память)   │  │ ├── docling-mcp (stdio)      │   │
│  │ embed:     │  │ └── docling-search (stdio)    │   │
│  │ text-embed-│  └──────┬───────────────────────┘   │
│  │ ding-3-small│         │                          │
│  └─────┬──────┘         │                          │
│        │                │                          │
└────────┼────────────────┼──────────────────────────┘
         │                │
         │     ┌──────────┘
         │     │
    ┌────┴─────┴───────┐
    │   Polz-proxy     │
    │   :8787          │
    │   chat & embed   │
    └──┬──────────┬────┘
       │          │
       │     ┌────┴──────┐
       │     │ ChromaDB  │
       │     │ ~/docling-│
       │     │ search/   │
       │     │ chroma/   │
       │     └────┬──────┘
       │          │
  ┌────┴──────────┴──────┐
  │    Scrinium :9231     │
  │    Веб-морда          │
  │    ────────           │
  │    Upload → docling    │
  │    (subprocess)        │
  │    → embed → ChromaDB  │
  │    Chat → Polz-proxy   │
  └──────────────────────┘
```

---

## 4. Пути и файлы

| Путь | Назначение |
|------|-----------|
| `~/.hermes/config.yaml` | Основной конфиг Hermes |
| `~/.hermes/config.addon.yaml` | MCP-серверы (docling) |
| `~/.hermes/.env` | Переменные окружения Hermes |
| `~/git/scrinium/` | Исходники веб-морды |
| `~/git/scrinium/.env` | Конфиг Scrinium |
| `~/git/scrinium/data/scrinium.db` | SQLite (чаты, сессии) |
| `~/git/scrinium/data/uploads/` | Загруженные файлы |
| `~/git/docling-mcp/` | Исходники docling-серверов |
| `~/git/docling-mcp/docling_search.py` | MCP-сервер поиска |
| `~/git/docling-mcp/mcp_wrapper.sh` | Обёртка для docling-mcp |
| `~/docling-search/chroma/` | ChromaDB data |
| `~/docling-search/index.json` | Реестр проиндексированных документов |

---

## 5. Запуск и управление

### Порядок запуска (ручной)
```bash
# 1. Polz-proxy (если не в systemd)
# уже должен быть запущен

# 2. Scrinium
cd ~/git/scrinium
source .venv/bin/activate
python3 -m scrinium    # → :9231

# 3. Hermes Agent сам поднимает MCP-серверы при старте
systemctl --user restart hermes   # если systemd
```

### Проверка здоровья
```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:9231/    # Scrinium
curl -s -o /dev/null -w '%{http_code}' http://localhost:8787     # Polz-proxy
curl -s -o /dev/null -w '%{http_code}' http://localhost:8642     # Hermes API
```

### Очистка ChromaDB (при смене модели)
```bash
python3 -c "
import chromadb
c = chromadb.PersistentClient(path=os.path.expanduser('~/docling-search/chroma'))
c.delete_collection('docs')
c.get_or_create_collection('docs', metadata={'hnsw:space': 'cosine'})
"
rm -f ~/docling-search/index.json
```

### OOM-защита
На сервере 8 GB RAM, docling при конвертации PDF жрёт ~2+ GB.
Добавлен swap 6 GB:
```bash
# swapfile уже добавлен
swapon --show   # 6.0 GB total
```

---

## 6. Безопасность

**Что в открытых репозиториях:**
- `akrhin/scrinium` — публичный. Ключей и паролей нет (в .gitignore)
- `akrhin/docling-mcp` — приватный. Ключи через env, в коде `""`

**Где лежат секреты:**
| Секрет | Где хранится |
|--------|-------------|
| ADMIN_PASSWORD | `.env` (в .gitignore) |
| Polz-proxy API key | `~/.hermes/.env` + `.env` локально |
| LLM_API_KEY | `.env` (в .gitignore) |
| EMBEDDING_API_KEY | `~/.hermes/config.addon.yaml` (локально) |

---

## 7. Зависимости

```
Scrinium:
├── FastAPI + Uvicorn (веб-сервер)
├── chromadb (векторная БД)
├── docling (конвертация PDF)
├── httpx (HTTP-клиент)
├── sqlalchemy + aiofiles (БД + файлы)
├── pyjwt + bcrypt (аутентификация)
├── python-dotenv (конфиг)
├── fastembed (запасной local embedder)
└── numpy (нормализация векторов)

Docling-search:
├── mcp (MCP-протокол)
├── chromadb (векторная БД)
├── httpx (HTTP-клиент для эмбеддингов)
└── numpy (нормализация)

Docling-mcp:
├── docling (официальный MCP-сервер)
├── torch (PyTorch для docling)
└── onnxruntime (RapidOCR для сканов)
```

---

## 8. История изменений

| Дата | Что изменилось |
|------|---------------|
| 2026-07-03 | Создан docling-mcp + docling-search (jina-embeddings-v3, local) |
| 2026-07-06 | Создан Scrinium — веб-интерфейс для загрузки и RAG-чата |
| 2026-07-06 | CORS, no-cache, upload progress |
| 2026-07-06 | Унификация: одна модель (text-embedding-3-small), одна коллекция (docs) |
| 2026-07-06 | Эмбеддинги через Polz-proxy (0 RAM), docling-search переведён на внешний API |
| 2026-07-06 | Swap 6 GB для защиты от OOM |
