"""
Scrinium — RAG Chat over your documents.

FastAPI web frontend. Shares ChromaDB with docling-search (MCP):
  • Same DB: ~/docling-search/chroma/
  • Same model: jinaai/jina-embeddings-v3 (fastembed)
  • Same collection: "docs"
  • What you upload via web → I find via MCP tools
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import uuid

log = logging.getLogger("scrinium")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import httpx
import jwt
import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Text, Integer, DateTime, ForeignKey,
    select, func, JSON as SQLJSON
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Config ─────────────────────────────────────────────────────────────

load_dotenv()

# ChromaDB — shared with docling-search MCP
CHROMA_DIR = os.path.expanduser(os.getenv("CHROMA_DIR", "~/docling-search/chroma"))
CHROMA_COLLECTION = "docs"

DB_PATH = Path(os.getenv("DB_PATH", "./data/scrinium.db"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    import hashlib
    JWT_SECRET = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
JWT_ALGO = "HS256"
JWT_TTL_HOURS = 24

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── Embedding provider ────────────────────��────────────────────

# local = fastembed + jina (бесплатно, ~700 MB RAM)
# remote = OpenAI-совместимый API (zero RAM, нужен ключ)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# Для remote-режима:
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "text-embedding-3-small")

_embedder = None  # fastembed (только для local)
_chroma = None

# Размерность векторов — важно для ChromaDB
# По умолчанию 1024 (jina), для внешних API может отличаться
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# ── Docling provider ─────────────────────────────────────────────

# local = запускать docling в подпроцессе (convert.py)
# remote = вызывать внешний HTTP API для конвертации
DOCLING_PROVIDER = os.getenv("DOCLING_PROVIDER", "local")

# Для remote-режима:
DOCLING_API_URL = os.getenv("DOCLING_API_URL", "").rstrip("/")
DOCLING_API_KEY = os.getenv("DOCLING_API_KEY", "")


def _get_embedder():
    """Lazy fastembed (только для local-режима)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v3")
    return _embedder


def _get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
    return _chroma


async def _embed_texts(texts: list[str], embed_type: str = "passage") -> list[list[float]]:
    """
    Заэмбедить список текстов.
    embed_type: "passage" (для индексации) или "query" (для поиска).
    В local-режиме это влияет на префикс в jina.
    В remote-режиме игнорируется.
    """
    import numpy as np

    if EMBEDDING_PROVIDER == "local":
        emb = _get_embedder()
        if embed_type == "query":
            vecs = list(emb.query_embed(texts[0] if len(texts) == 1 else texts))
        else:
            vecs = list(emb.embed(texts, embed_type="passage"))
        return [(np.array(v) / np.linalg.norm(v)).tolist() for v in vecs]

    elif EMBEDDING_PROVIDER == "remote":
        if not EMBEDDING_API_URL:
            raise RuntimeError("EMBEDDING_API_URL не указан для remote-режима")
        url = f"{EMBEDDING_API_URL}/embeddings"
        headers = {"Content-Type": "application/json"}
        if EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"
        async with httpx.AsyncClient(timeout=30) as cl:
            r = await cl.post(url, headers=headers, json={
                "model": EMBEDDING_API_MODEL,
                "input": texts,
                "dimensions": 1024,
            })
        if r.status_code != 200:
            raise RuntimeError(f"Embedding API error {r.status_code}: {r.text[:200]}")

        data = r.json()
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return [np.array(v / np.linalg.norm(v)).tolist() for v in embeddings]

    else:
        raise RuntimeError(f"Неизвестный EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


# ── Database ────────────────────────────────────────────────────────────

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="New chat")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources = Column(SQLJSON, nullable=True)
    created_at = Column(DateTime, default=func.now())


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    doc_id = Column(String(64), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pages = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


Base.metadata.create_all(engine)

# ── Auth ────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def _hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _check_pw(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def _create_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "iat": now, "exp": now + timedelta(hours=JWT_TTL_HOURS)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def _verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def _get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _verify_token(credentials.credentials)


# ── LLM helper ─────────────────────────────────────────────────────────

async def _llm_ask(system: str, messages: list[dict]) -> str:
    if not LLM_BASE_URL or not LLM_API_KEY:
        return "LLM не настроен. Укажите LLM_BASE_URL, LLM_MODEL и LLM_API_KEY в .env"
    full = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=60) as cl:
        r = await cl.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "messages": full, "temperature": 0.3},
        )
    if r.status_code != 200:
        return f"LLM error {r.status_code}: {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"]


# ── Document processing ────────────────────────────────────────────────

_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".html", ".htm", ".json", ".csv",
                  ".xml", ".yaml", ".yml", ".rst", ".rtf", ".epub",
                  ".docx", ".xlsx", ".pptx"}

CHUNK_MIN = 200
CHUNK_MAX = 2000


def _chunk_text(text: str, source: str) -> list[tuple[str, str]]:
    """Chunk by paragraphs — same logic as docling-search."""
    import re
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    current = ""
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < CHUNK_MAX:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunk_id = f"{source}#ch{idx}"
                chunks.append((current, chunk_id))
                idx += 1
            current = para
    if current:
        chunk_id = f"{source}#ch{idx}"
        chunks.append((current, chunk_id))
    return chunks


async def _process_file(filepath: Path, filename: str) -> tuple[str, int, str]:
    """Convert file to markdown using docling, return (markdown, pages, doc_id)."""
    doc_id = f"{uuid.uuid4().hex[:12]}"
    ext = Path(filename).suffix.lower()

    if ext in {".txt", ".md", ".html", ".htm", ".json", ".csv", ".xml", ".yaml", ".yml", ".rst"}:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = await f.read()
        return text, 0, doc_id

    return await _run_docling(filepath, doc_id)

async def _run_docling(filepath: Path, doc_id: str) -> tuple[str, int, str]:
    """Convert document to markdown. Local = subprocess, Remote = HTTP API."""
    if DOCLING_PROVIDER == "remote":
        return await _run_docling_remote(filepath, doc_id)

    # Local: subprocess через convert.py
    convert_script = Path(__file__).parent / "convert.py"

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(convert_script), str(filepath),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3600)
    except asyncio.TimeoutError:
        proc.kill()
        raise asyncio.TimeoutError("Конвертация не завершилась за 60 минут — файл слишком большой")

    if proc.returncode == -9:
        raise RuntimeError("docling убит (OOM — не хватило памяти для конвертации PDF)")
    if proc.returncode != 0:
        raise RuntimeError(f"docling завершился с кодом {proc.returncode}: {stderr.decode()[:500]}")

    result = json.loads(stdout.decode())
    if "error" in result:
        raise RuntimeError(result["error"])

    return result["markdown"], result["pages"], doc_id


async def _run_docling_remote(filepath: Path, doc_id: str) -> tuple[str, int, str]:
    """Convert file via external HTTP API (e.g. docling-serve running elsewhere)."""
    if not DOCLING_API_URL:
        raise RuntimeError("DOCLING_API_URL не у��азан для remote-режима")

    async with aiofiles.open(filepath, "rb") as f:
        content = await f.read()

    filename = filepath.name
    files = {"file": (filename, content, "application/octet-stream")}

    headers = {}
    if DOCLING_API_KEY:
        headers["Authorization"] = f"Bearer {DOCLING_API_KEY}"

    url = f"{DOCLING_API_URL}/convert"
    async with httpx.AsyncClient(timeout=300) as cl:
        r = await cl.post(url, files=files, headers=headers if headers else None)

    if r.status_code != 200:
        raise RuntimeError(f"docling remote error {r.status_code}: {r.text[:300]}")

    data = r.json()
    return data.get("markdown", ""), data.get("pages", 0), doc_id


async def _index_document(text: str, doc_id: str, filename: str) -> int:
    """Chunk + embed → ChromaDB (shared with docling-search)."""
    col = _get_chroma()
    chunks = _chunk_text(text, doc_id)

    texts = [c[0] for c in chunks]
    ids = [c[1] for c in chunks]
    metadatas = [{"doc_id": doc_id, "filename": filename, "source": cid} for cid in ids]

    BATCH = 20
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i : i + BATCH]
        batch_ids = ids[i : i + BATCH]
        batch_meta = metadatas[i : i + BATCH]
        vecs = await _embed_texts(batch_texts, embed_type="passage")
        col.add(
            ids=batch_ids,
            embeddings=vecs,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    return len(chunks)


async def _search_chroma(query: str, top_k: int = 5) -> list[dict]:
    """Embed query → search ChromaDB (cosine, normalized)."""
    col = _get_chroma()
    vecs = await _embed_texts([query], embed_type="query")

    try:
        results = col.query(query_embeddings=vecs, n_results=top_k)
    except Exception:
        return []

    out = []
    if not results["ids"] or not results["ids"][0]:
        return out

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        out.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i][:2000] if results["documents"] else "",
            "score": round(results["distances"][0][i], 4) if results["distances"] else 0,
            "filename": meta.get("filename", "unknown"),
            "doc_id": meta.get("doc_id", ""),
        })
    return out


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="Scrinium", version="0.1.0")

# CORS — allow access from any browser (phone, laptop, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import starlette.datastructures
app.file_max_size = 100 * 1024 * 1024


@app.on_event("startup")
async def startup():
    db = SessionLocal()
    try:
        existing = db.execute(select(User).where(User.username == ADMIN_USER)).scalar_one_or_none()
        if existing is None:
            db.add(User(username=ADMIN_USER, password_hash=_hash_pw(ADMIN_PASSWORD)))
        else:
            if not _check_pw(ADMIN_PASSWORD, existing.password_hash):
                existing.password_hash = _hash_pw(ADMIN_PASSWORD)
        db.commit()
    finally:
        db.close()

    # Preload embedder (в local-режиме)
    if EMBEDDING_PROVIDER == "local":
        _get_embedder()


# ── API Routes ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def api_login(req: LoginRequest):
    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == req.username)).scalar_one_or_none()
        if user is None or not _check_pw(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = _create_token(user.username)
        return {"token": token, "username": user.username}
    finally:
        db.close()


@app.get("/api/me")
async def api_me(username: str = Depends(_get_current_user)):
    return {"username": username}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    username: str = Depends(_get_current_user),
):
    """Upload a file → docling → ChromaDB (shared with docling-search MCP)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")

    ext = Path(file.filename).suffix.lower()
    if ext not in _SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    log.info("Upload received: %s (%d bytes)", file.filename, len(content))
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)

    try:
        markdown, pages, doc_id = await _process_file(filepath, file.filename)
    except RuntimeError as e:
        log.error("Docling conversion failed for %s: %s", file.filename, e)
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Ошибка конвертации: {e}")
    except asyncio.TimeoutError:
        log.error("Docling conversion timed out for %s", file.filename)
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Конвертация не завершилась за 5 минут. Файл слишком большой или повреждён.")
    except Exception as e:
        log.error("Unexpected conversion error for %s: %s", file.filename, e)
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Неизвестная ошибка конвертации: проверьте логи")

    log.info("Doc converted: %s → %d chars, %d pages, id=%s", file.filename, len(markdown), pages, doc_id)

    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == username)).scalar_one()
        chunks = await _index_document(markdown, doc_id, file.filename)
        doc = Document(
            filename=file.filename,
            filepath=str(filepath),
            doc_id=doc_id,
            user_id=user.id,
            pages=pages,
            chunks=chunks,
        )
        db.add(doc)
        db.commit()
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": pages,
            "chunks": chunks,
            "message": f"Indexed {chunks} chunks from {file.filename}",
        }
    finally:
        db.close()


@app.get("/api/documents")
async def api_documents(username: str = Depends(_get_current_user)):
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Document).join(User).where(User.username == username)
            .order_by(Document.created_at.desc())
        ).scalars().all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "doc_id": d.doc_id,
                "pages": d.pages,
                "chunks": d.chunks,
                "created_at": d.created_at.isoformat(),
            }
            for d in rows
        ]
    finally:
        db.close()


@app.delete("/api/documents/{doc_id}")
async def api_delete_document(doc_id: str, username: str = Depends(_get_current_user)):
    db = SessionLocal()
    try:
        doc = db.execute(
            select(Document).where(Document.doc_id == doc_id)
        ).scalar_one_or_none()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        db.delete(doc)
        _get_chroma().delete(where={"doc_id": doc_id})
        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/api/chat")
async def api_chat(
    req: ChatRequest,
    username: str = Depends(_get_current_user),
):
    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == username)).scalar_one()

        if req.session_id:
            session = db.execute(
                select(ChatSession).where(
                    ChatSession.id == req.session_id,
                    ChatSession.user_id == user.id,
                )
            ).scalar_one_or_none()
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = ChatSession(
                id=uuid.uuid4().hex[:12],
                user_id=user.id,
                title=req.message[:80],
            )
            db.add(session)
            db.commit()

        db.add(Message(session_id=session.id, role="user", content=req.message))

        sources = await _search_chroma(req.message, top_k=5)

        context_parts = []
        doc_map = {}
        for s in sources:
            fn = s["filename"]
            if fn not in doc_map:
                doc_map[fn] = []
            doc_map[fn].append(s["text"])

        for fn, texts in doc_map.items():
            context_parts.append(f"--- {fn} ---\n" + "\n\n".join(texts[:3]))

        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        system_prompt = (
            "You are a helpful RAG assistant. Answer based on the provided context. "
            "If the context does not contain the answer, say so. "
            "Always cite which document(s) you used. Ответь на русском языке."
        )

        has_context = bool(context_parts)
        llm_messages = []
        if has_context:
            llm_messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {req.message}",
            })
        else:
            previous = db.execute(
                select(Message).where(Message.session_id == session.id)
                .order_by(Message.created_at.desc()).limit(6)
            ).scalars().all()
            for m in reversed(previous):
                llm_messages.append({"role": m.role, "content": m.content})
            llm_messages.append({"role": "user", "content": req.message})

        answer = await _llm_ask(system_prompt, llm_messages)

        msg = Message(
            session_id=session.id,
            role="assistant",
            content=answer,
            sources=[s["id"] for s in sources] if sources else None,
        )
        db.add(msg)
        db.commit()

        return {
            "session_id": session.id,
            "answer": answer,
            "sources": sources,
        }
    finally:
        db.close()


@app.get("/api/chat/sessions")
async def api_chat_sessions(username: str = Depends(_get_current_user)):
    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == username)).scalar_one()
        rows = db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.updated_at.desc())
        ).scalars().all()
        return [
            {"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()}
            for s in rows
        ]
    finally:
        db.close()


@app.get("/api/chat/sessions/{session_id}")
async def api_chat_session_messages(
    session_id: str,
    username: str = Depends(_get_current_user),
):
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        ).scalars().all()
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "created_at": m.created_at.isoformat(),
            }
            for m in rows
        ]
    finally:
        db.close()


@app.delete("/api/chat/sessions/{session_id}")
async def api_delete_session(session_id: str, username: str = Depends(_get_current_user)):
    db = SessionLocal()
    try:
        db.execute(Message.__table__.delete().where(Message.session_id == session_id))
        db.execute(ChatSession.__table__.delete().where(ChatSession.id == session_id))
        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()


class LLMSettings(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""


@app.get("/api/llm/settings")
async def api_llm_settings_get(username: str = Depends(_get_current_user)):
    return {
        "base_url": LLM_BASE_URL,
        "model": LLM_MODEL,
        "api_key": f"{LLM_API_KEY[:6]}..." if LLM_API_KEY else "",
    }


# ── Frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def serve_frontend():
    template = Path(__file__).parent / "templates" / "index.html"
    if not template.exists():
        return HTMLResponse("<h1>Scrinium</h1><p>Frontend not found.</p>")
    html = template.read_text(encoding="utf-8")
    # Cache-bust: add timestamp so browser never caches old HTML
    ts = datetime.now().strftime("%Y%m%d%H%M")
    html = html.replace("</head>", f'<meta name="version" content="{ts}"></head>')
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    return HTMLResponse(html, headers=headers)


# ── Entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9231"))
    uvicorn.run("scrinium:app", host="0.0.0.0", port=port, reload=False)
