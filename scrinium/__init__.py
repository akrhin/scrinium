"""
Scrinium — RAG Chat over your documents.

FastAPI backend with:
  • JWT auth (multiple users)
  • File upload → docling-mcp → chunk → ChromaDB
  • Chat: semantic search + LLM answer
  • Chat history (sessions)
  • Responsive single-page frontend
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import chromadb
import httpx
import jwt
import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Text, Integer, Float, DateTime, ForeignKey,
    select, func, JSON as SQLJSON
)
from sqlalchemy.orm import declarative_base, Session as SASession, sessionmaker

# ── Config ─────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
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

USERS = {}  # populated at startup

CHROMA_COLLECTION = "scrinium"
DOCLING_SCRIPT = os.getenv("DOCLING_SCRIPT", "")

# Ensure dirs
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

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

# ── ChromaDB ────────────────────────────────────────────────────────────

_chroma_client: chromadb.ClientAPI | None = None
_chroma_collection = None


def _get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    return _chroma_collection


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

_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".html", ".htm", ".json", ".csv", ".xml", ".yaml", ".yml", ".rst", ".rtf", ".epub", ".docx", ".xlsx", ".pptx"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _chunk_text(text: str, source: str) -> list[tuple[str, str]]:
    """Return list of (chunk_text, chunk_id)"""
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunk_id = f"{source}#chunk{len(chunks)}"
        chunks.append((chunk, chunk_id))
        i += CHUNK_SIZE - CHUNK_OVERLAP
        if i >= len(words):
            break
    return chunks


async def _process_file(filepath: Path, filename: str) -> tuple[str, int, str]:
    """Convert file to markdown using docling, return (markdown, pages, doc_id)"""
    doc_id = f"{uuid.uuid4().hex[:12]}"

    ext = Path(filename).suffix.lower()

    # Plain text — read directly
    if ext in {".txt", ".md", ".html", ".htm", ".json", ".csv", ".xml", ".yaml", ".yml", ".rst"}:
        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = await f.read()
        return text, 0, doc_id

    # Binary/document formats — use docling
    return await _run_docling(filepath, doc_id)


async def _run_docling(filepath: Path, doc_id: str) -> tuple[str, int, str]:
    """Run docling on a binary document. Runs in thread to avoid blocking."""
    import asyncio
    from docling.document_converter import DocumentConverter

    def _convert():
        converter = DocumentConverter()
        doc = converter.convert(str(filepath))
        md = doc.document.export_to_markdown()
        pages = len(doc.pages) if hasattr(doc, 'pages') else 0
        return md, pages

    loop = asyncio.get_event_loop()
    markdown, pages = await loop.run_in_executor(None, _convert)
    return markdown, pages, doc_id


async def _index_document(text: str, doc_id: str, filename: str, user_id: int) -> int:
    """Chunk text and index into ChromaDB"""
    collection = _get_chroma()
    chunks = _chunk_text(text, doc_id)

    metadatas = []
    ids = []
    documents = []

    for chunk_text, chunk_id in chunks:
        documents.append(chunk_text)
        ids.append(chunk_id)
        metadatas.append({
            "doc_id": doc_id,
            "filename": filename,
            "source": chunk_id,
        })

    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    return len(chunks)


async def _search_chroma(query: str, top_k: int = 5) -> list[dict]:
    """Search ChromaDB and return results with source info"""
    collection = _get_chroma()
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
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


@app.on_event("startup")
async def startup():
    # Ensure admin user exists
    db = SessionLocal()
    try:
        existing = db.execute(select(User).where(User.username == ADMIN_USER)).scalar_one_or_none()
        if existing is None:
            db.add(User(username=ADMIN_USER, password_hash=_hash_pw(ADMIN_PASSWORD)))
            db.commit()
    finally:
        db.close()

    # Ensure ChromaDB collection exists
    _get_chroma()


# ── API Routes ──────────────────────────────────────────────────────────

# --- Auth ---

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


# --- Documents ---

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    username: str = Depends(_get_current_user),
):
    """Upload a file → docling → ChromaDB"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")

    ext = Path(file.filename).suffix.lower()
    if ext not in _SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)

    markdown, pages, doc_id = await _process_file(filepath, file.filename)

    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == username)).scalar_one()
        chunks = await _index_document(markdown, doc_id, file.filename, user.id)
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

        # Remove from ChromaDB
        collection = _get_chroma()
        collection.delete(where={"doc_id": doc_id})

        db.commit()
        return {"message": "Deleted"}
    finally:
        db.close()


# --- Chat ---

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

        # Resolve session
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

        # Save user message
        db.add(Message(session_id=session.id, role="user", content=req.message))

        # Search
        sources = await _search_chroma(req.message, top_k=5)

        # Build context
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

        # Save assistant message
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


# --- LLM settings (per-user in-memory — can extend to DB later) ---

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
        return HTMLResponse("<h1>Scrinium</h1><p>Frontend not found. Run from project root.</p>")
    return HTMLResponse(template.read_text(encoding="utf-8"))


# ── Entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", "9231"))
    uvicorn.run("scrinium.__main__:app", host="0.0.0.0", port=PORT, reload=True)
