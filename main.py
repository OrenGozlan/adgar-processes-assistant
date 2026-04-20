import os
import re
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, init_db
from models import User, Document, Chunk, ChatSession, Message, TopQuestion
from auth import (
    hash_password, verify_password, create_token,
    get_current_user, require_admin,
)
from embeddings import embed_text, embed_texts, cosine_similarity, is_model_ready
from document_parser import extract_text, chunk_text

app = FastAPI(title="Adgar's Processes Personal Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "/data/uploads" if os.path.isdir("/data") else "./uploads"


@app.on_event("startup")
def startup():
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    import threading
    threading.Thread(target=_startup_background, daemon=True).start()


def _startup_background():
    from embeddings import _get_model
    _get_model()
    _retry_stuck_documents()


def _retry_stuck_documents():
    db = next(get_db())
    try:
        stuck = db.query(Document).filter(Document.status == "processing").all()
        for doc in stuck:
            filepath = os.path.join(UPLOAD_DIR, doc.filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    file_bytes = f.read()
                _process_document(doc.id, file_bytes, doc.original_name)
            else:
                doc.status = "error"
                db.commit()
    finally:
        db.close()


# --------------- Health ---------------

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": is_model_ready()}


# --------------- Auth ---------------

class RegisterBody(BaseModel):
    email: str
    password: str
    role: str = "employee"
    admin_code: str | None = None


class LoginBody(BaseModel):
    email: str
    password: str


ALLOWED_EMAIL_DOMAINS = [r"adgar\..+", r"greems\.io"]


def _is_email_domain_allowed(email: str) -> bool:
    domain = email.split("@")[-1].lower()
    return any(re.fullmatch(pattern, domain) for pattern in ALLOWED_EMAIL_DOMAINS)


@app.post("/api/auth/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    if not _is_email_domain_allowed(body.email):
        raise HTTPException(403, "Registration is restricted to authorized company domains")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    role = "employee"
    if body.role == "admin":
        expected = os.getenv("ADMIN_REGISTRATION_CODE", "")
        if not expected or body.admin_code != expected:
            raise HTTPException(403, "Invalid admin registration code")
        role = "admin"
    user = User(email=body.email, hashed_password=hash_password(body.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email, "role": user.role, "token": create_token(user.id, user.role)}


@app.post("/api/auth/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    return {"id": user.id, "email": user.email, "role": user.role, "token": create_token(user.id, user.role)}


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


# --------------- Admin: Documents ---------------

def _process_document(doc_id: int, file_bytes: bytes, filename: str):
    import threading
    db = next(get_db())
    try:
        text = extract_text(file_bytes, filename)
        if not text.strip():
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "error"
                db.commit()
            return

        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)

        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return

        for i, (chunk_text_item, emb) in enumerate(zip(chunks, embeddings)):
            c = Chunk(document_id=doc.id, content=chunk_text_item, chunk_index=i)
            c.set_embedding(emb)
            db.add(c)

        doc.chunk_count = len(chunks)
        doc.status = "ready"
        db.commit()
    except Exception:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "error"
            db.commit()
    finally:
        db.close()


@app.post("/api/admin/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    allowed = (".pdf", ".docx", ".txt")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    content = await file.read()

    stored_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, stored_name)
    with open(filepath, "wb") as f:
        f.write(content)
    doc = Document(
        filename=stored_name,
        original_name=file.filename,
        uploaded_by=user.id,
        chunk_count=0,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    import threading
    threading.Thread(
        target=_process_document,
        args=(doc.id, content, file.filename),
        daemon=True,
    ).start()

    return {
        "id": doc.id,
        "filename": doc.original_name,
        "status": "processing",
        "upload_date": doc.upload_date.isoformat(),
    }


@app.get("/api/admin/documents")
def list_documents(user: User = Depends(require_admin), db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.original_name,
            "upload_date": d.upload_date.isoformat(),
            "chunk_count": d.chunk_count,
            "active": d.active,
            "status": d.status or "ready",
        }
        for d in docs
    ]


@app.delete("/api/admin/documents/{doc_id}")
def delete_document(doc_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    db.delete(doc)
    db.commit()
    return {"detail": "Document deleted"}


@app.patch("/api/admin/documents/{doc_id}/toggle")
def toggle_document(doc_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    doc.active = not doc.active
    db.commit()
    return {"id": doc.id, "active": doc.active}


# --------------- Chat ---------------

LANGUAGE_MAP = {"en": "English", "he": "Hebrew (עברית)", "pl": "Polish (Polski)"}


class ChatMessageBody(BaseModel):
    session_id: int | None = None
    message: str
    language: str = "en"


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _update_top_questions(db: Session, question: str):
    normalized = _normalize_question(question)
    existing = db.query(TopQuestion).filter(TopQuestion.question_text == normalized).first()
    if existing:
        existing.count += 1
        existing.last_asked = datetime.now(timezone.utc)
    else:
        db.add(TopQuestion(question_text=normalized, count=1, last_asked=datetime.now(timezone.utc)))
    db.commit()


def _find_relevant_chunks(db: Session, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
    active_doc_ids = [d.id for d in db.query(Document.id).filter(Document.active == True).all()]
    if not active_doc_ids:
        return []
    chunks = db.query(Chunk).filter(Chunk.document_id.in_(active_doc_ids)).all()
    scored = []
    for chunk in chunks:
        sim = cosine_similarity(query_embedding, chunk.get_embedding())
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


SYSTEM_PROMPT_TEMPLATE = (
    "You are Adgar's Processes Personal Assistant. You help employees understand company "
    "procedures and guidelines.\n\n"
    "Rules:\n"
    "- Answer ONLY based on the provided context. If the answer is not in the documents, say so honestly. Never make up procedures.\n"
    "- Be VERY concise — 1-3 sentences max. Give the direct answer, not background. Use bullet points only when listing steps.\n"
    "- You MUST reply in: {language}.\n"
    "- If replying in Hebrew: you MUST write like a native Israeli talks at work. This is the #1 priority.\n"
    "  GOOD Hebrew examples:\n"
    "  - \"צריך להגיש את הטופס דרך הפורטל ואז לשלוח מייל למנהל שלך\"\n"
    "  - \"אפשר לבקש את זה מהמשרד, פשוט תשלח מייל\"\n"
    "  - \"קודם כל תבדוק אם יש לך הרשאה, ואז תיכנס למערכת\"\n"
    "  BAD Hebrew (NEVER write like this):\n"
    "  - \"יש צורך בהגשת הטופס באמצעות הפורטל\" ← too formal\n"
    "  - \"ניתן לפנות אל המשרד לצורך קבלת\" ← sounds like a government form\n"
    "  - \"על מנת לבצע את הפעולה, יש לוודא כי\" ← robotic\n"
    "  Key rules: 'צריך' not 'יש צורך', 'אפשר' not 'ניתן', 'בגלל' not 'מאחר', "
    "'כדי' not 'על מנת', 'לגבי' not 'באשר ל', 'אם' not 'במידה ו', "
    "'עדיף' not 'מומלץ', 'תשלח/תבדוק' not 'יש לשלוח/יש לבדוק'.\n"
    "  Use second person (אתה/את) not impersonal. Short sentences. No fancy words.\n"
    "- If replying in Polish: use everyday conversational Polish.\n"
    "- IMPORTANT: Your response MUST end with a line containing exactly `---` followed by a short follow-up suggestion "
    "(a question to dig deeper, or suggest who to contact). "
    "This follow-up line will be displayed as a separate clickable bubble. Write it in the same language as the answer.\n"
    "- Never switch languages mid-answer."
)


@app.post("/api/chat/message")
def chat_message(
    body: ChatMessageBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import anthropic

    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    if not is_model_ready():
        raise HTTPException(503, "The system is still loading. Please try again in a moment.")

    if body.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == body.session_id, ChatSession.user_id == user.id
        ).first()
        if not session:
            raise HTTPException(404, "Session not found")
    else:
        title = body.message[:80].strip()
        session = ChatSession(user_id=user.id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)

    user_msg = Message(session_id=session.id, role="user", content=body.message)
    db.add(user_msg)
    db.commit()

    query_emb = embed_text(body.message)
    relevant_chunks = _find_relevant_chunks(db, query_emb)

    context_text = ""
    if relevant_chunks:
        context_parts = [f"[Document chunk {i+1}]\n{c.content}" for i, c in enumerate(relevant_chunks)]
        context_text = "\n\n---\n\n".join(context_parts)

    user_content = body.message
    if context_text:
        user_content = f"Context from company documents:\n\n{context_text}\n\n---\n\nEmployee question: {body.message}"

    history = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at)
        .all()
    )
    messages = []
    for msg in history[:-1]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_content})

    _update_top_questions(db, body.message)

    lang_name = LANGUAGE_MAP.get(body.language, "English")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=lang_name)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    session_id = session.id

    def generate():
        full_response = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield f"data: {text}\n\n"

        assistant_msg = Message(session_id=session_id, role="assistant", content=full_response)
        db_inner = next(get_db())
        try:
            db_inner.add(assistant_msg)
            db_inner.commit()
        finally:
            db_inner.close()

        yield f"event: done\ndata: {session_id}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Session-Id": str(session_id)},
    )


@app.get("/api/chat/sessions")
def list_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc()).all()
    result = []
    for s in sessions:
        msg_count = db.query(func.count(Message.id)).filter(Message.session_id == s.id).scalar()
        result.append({
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "message_count": msg_count,
        })
    return result


@app.get("/api/chat/sessions/{session_id}/messages")
def get_session_messages(session_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
    return [
        {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in messages
    ]


@app.delete("/api/chat/sessions/{session_id}")
def delete_session(session_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return {"detail": "Session deleted"}


@app.get("/api/chat/top-questions")
def top_questions(db: Session = Depends(get_db)):
    questions = db.query(TopQuestion).order_by(TopQuestion.count.desc()).limit(5).all()
    return [{"question": q.question_text, "count": q.count} for q in questions]


# --------------- Static frontend ---------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
