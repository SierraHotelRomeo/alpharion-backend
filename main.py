import os
import re
import math
import hmac
import base64
import json
import secrets
import sqlite3
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import requests
import yfinance as yf
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except Exception:
    FDR_AVAILABLE = False


# =========================================================
# Auth / Runtime Settings
# =========================================================
# 회원정보 DB 저장 경로
# Render Persistent Disk를 사용하는 경우 Environment Variable에 아래처럼 설정하세요.
# AUTH_DB_PATH=/var/data/alpharion_auth.db
#
# 기존에 /opt/render/project/src/alpharion_auth.db에 있던 DB가 있고,
# /var/data/alpharion_auth.db가 아직 없으면 서버 시작 시 자동으로 1회 복사합니다.
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH") or os.getenv("DB_PATH") or "/var/data/alpharion_auth.db"
LEGACY_AUTH_DB_PATH = "/opt/render/project/src/alpharion_auth.db"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_SECRET_KEY_ON_RENDER")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))
CAPTCHA_EXPIRE_MINUTES = int(os.getenv("CAPTCHA_EXPIRE_MINUTES", "10"))

# Render 배포 주소와 Netlify 프론트 주소를 본인 환경에 맞게 설정하세요.
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://alpharion.cloud")
API_PUBLIC_BASE = os.getenv("API_PUBLIC_BASE", "https://alpharion-backend.onrender.com")


# =========================================================
# Payment / Plan Settings
# =========================================================
STANDARD_PRICE_KRW = int(os.getenv("STANDARD_PRICE_KRW", "2000"))
STANDARD_PLAN_DAYS = int(os.getenv("STANDARD_PLAN_DAYS", "31"))
FREE_AI_LIMIT = int(os.getenv("FREE_AI_LIMIT", "5"))

# NICEPAY v1 JavaScript 결제 연동용 환경변수
# Render Environment Variables에 아래 값을 설정하세요.
# NICEPAY_CLIENT_ID=...
# NICEPAY_SECRET_KEY=...
NICEPAY_CLIENT_ID = os.getenv("NICEPAY_CLIENT_ID", "")
NICEPAY_SECRET_KEY = os.getenv("NICEPAY_SECRET_KEY", "")
NICEPAY_APPROVE_URL = "https://api.nicepay.co.kr/v1/payments"

# =========================================================
# Admin Settings
# =========================================================
# Render Environment Variables에 ADMIN_SECRET_KEY를 추가하세요.
# admin.html에서 입력한 관리자 키와 이 값이 일치해야 회원 목록을 볼 수 있습니다.
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "")

# =========================================================
# Brevo Transactional Email API Settings
# =========================================================
# BREVO_API_KEY는 코드에 직접 넣지 않고 Render Environment Variables에서 불러옵니다.
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
BREVO_FROM_EMAIL = os.getenv("BREVO_FROM_EMAIL", "codegeneva@naver.com")
BREVO_FROM_NAME = os.getenv("BREVO_FROM_NAME", "Alpharion AI Market Watch")
PASSWORD_FIND_CODE_EXPIRE_MINUTES = int(os.getenv("PASSWORD_FIND_CODE_EXPIRE_MINUTES", "10"))

# bcrypt 72-byte 문제를 피하기 위해 pbkdf2_sha256 사용
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    password_confirm: str
    captcha_token: str
    captcha_answer: str
    terms_accepted: bool
    signup_verification_code: str



class SignupEmailCodeRequest(BaseModel):
    email: EmailStr
    captcha_token: str
    captcha_answer: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PasswordResetConfirmRequest(BaseModel):
    email: EmailStr
    recovery_code: str
    new_password: str
    new_password_confirm: str
    captcha_token: str
    captcha_answer: str


class PasswordFindRequest(BaseModel):
    email: EmailStr
    captcha_token: str
    captcha_answer: str


class PasswordFindConfirmRequest(BaseModel):
    email: EmailStr
    verification_code: str


class AdminSetStandardRequest(BaseModel):
    days: Optional[int] = None
    paid_at: Optional[str] = None
    reset_free_count: bool = True


class AdminSetFreeRequest(BaseModel):
    reset_free_count: bool = False


class NicepayPrepareRequest(BaseModel):
    pass


class ScreenerFilterItem(BaseModel):
    type: str
    key: str


class ScreenerFilterGroup(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    operator: str = "AND"
    filters: List[ScreenerFilterItem] = Field(default_factory=list)


class ScreenerRequest(BaseModel):
    market: str = "ALL"
    keyword: str = ""
    limit: int = 40
    patterns: List[str] = Field(default_factory=list)
    financials: List[str] = Field(default_factory=list)
    group_operator: str = "AND"
    groups: List[ScreenerFilterGroup] = Field(default_factory=list)


def validate_runtime_config():
    if JWT_SECRET_KEY == "CHANGE_THIS_SECRET_KEY_ON_RENDER":
        raise RuntimeError("JWT_SECRET_KEY not configured. Set JWT_SECRET_KEY in Render Environment Variables.")


def utcnow():
    return datetime.utcnow()


def get_auth_db():
    db_dir = os.path.dirname(AUTH_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate_legacy_auth_db_if_needed():
    """
    Render 재배포 후에도 회원정보가 유지되도록 Persistent Disk 경로를 사용합니다.
    기존 임시 경로 DB가 있고 새 Persistent Disk DB가 아직 없으면 최초 1회 자동 복사합니다.
    이미 /var/data/alpharion_auth.db가 있으면 절대 덮어쓰지 않습니다.
    """
    target_path = os.path.abspath(AUTH_DB_PATH)
    legacy_path = os.path.abspath(LEGACY_AUTH_DB_PATH)

    if target_path == legacy_path:
        return

    if os.path.exists(target_path):
        return

    if not os.path.exists(legacy_path):
        return

    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    try:
        shutil.copy2(legacy_path, target_path)
        print(f"AUTH DB migrated from {legacy_path} to {target_path}")
    except Exception as e:
        print("AUTH DB MIGRATION ERROR:", repr(e))




def ensure_column(conn, table_name: str, column_name: str, column_sql: str):
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {col["name"] for col in cols}
    if column_name not in existing:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def init_auth_db():
    conn = get_auth_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            recovery_code_hash TEXT,
            terms_accepted INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signup_email_verifications (
            email TEXT PRIMARY KEY,
            code_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # 기존 서버 DB를 그대로 쓰는 경우를 위한 안전 마이그레이션
    ensure_column(conn, "users", "recovery_code_hash", "recovery_code_hash TEXT")
    ensure_column(conn, "users", "terms_accepted", "terms_accepted INTEGER DEFAULT 0")
    ensure_column(conn, "users", "password_plain", "password_plain TEXT")
    ensure_column(conn, "users", "find_code_hash", "find_code_hash TEXT")
    ensure_column(conn, "users", "find_code_expires_at", "find_code_expires_at TEXT")

    # Standard 유료회원 / Free 사용횟수 관리용 컬럼
    ensure_column(conn, "users", "plan_type", "plan_type TEXT DEFAULT 'FREE'")
    ensure_column(conn, "users", "plan_started_at", "plan_started_at TEXT")
    ensure_column(conn, "users", "plan_expire_at", "plan_expire_at TEXT")
    ensure_column(conn, "users", "standard_paid_at", "standard_paid_at TEXT")
    ensure_column(conn, "users", "ai_analysis_count", "ai_analysis_count INTEGER DEFAULT 0")
    ensure_column(conn, "users", "free_ai_limit", f"free_ai_limit INTEGER DEFAULT {FREE_AI_LIMIT}")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payment_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            status TEXT NOT NULL,
            goods_name TEXT DEFAULT 'Alpharion Standard 1개월 이용권',
            auth_token TEXT,
            tid TEXT,
            raw_prepare TEXT,
            raw_approve TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # 기존 payment_orders 테이블이 이미 만들어진 경우에도 새 컬럼을 자동 추가합니다.
    # Render Persistent Disk의 SQLite DB는 재배포 후에도 유지되므로 CREATE TABLE IF NOT EXISTS만으로는
    # 기존 테이블 구조가 바뀌지 않습니다. 따라서 결제 오류 방지를 위해 ALTER TABLE 마이그레이션이 필요합니다.
    ensure_column(conn, "payment_orders", "goods_name", "goods_name TEXT DEFAULT 'Alpharion Standard 1개월 이용권'")
    ensure_column(conn, "payment_orders", "auth_token", "auth_token TEXT")
    ensure_column(conn, "payment_orders", "tid", "tid TEXT")
    ensure_column(conn, "payment_orders", "raw_prepare", "raw_prepare TEXT")
    ensure_column(conn, "payment_orders", "raw_approve", "raw_approve TEXT")

    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_runtime_config()
    migrate_legacy_auth_db_if_needed()
    init_auth_db()
    print(f"AUTH_DB_PATH={AUTH_DB_PATH}")
    yield


app = FastAPI(title="Alpharion Market Watch API", lifespan=lifespan)

allowed_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "https://alpharion.cloud,https://www.alpharion.cloud,http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# Auth Helpers
# ======================================================
def hash_secret(value: str):
    return pwd_context.hash(value)


def verify_secret(value: str, hashed_value: str):
    if not hashed_value:
        return False
    return pwd_context.verify(value, hashed_value)


def create_access_token(user_id: int, email: str):
    expire = utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "email": email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def make_recovery_code():
    # 예: AMW-A1B2-C3D4-E5F6
    raw = secrets.token_hex(6).upper()
    return f"AMW-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"


def make_email_verification_code():
    # 6자리 숫자 인증번호
    return str(secrets.randbelow(900000) + 100000)


def send_mail(to_email: str, subject: str, body: str):
    if not BREVO_API_KEY:
        raise HTTPException(status_code=500, detail="Brevo API Key가 설정되지 않았습니다. Render 환경변수 BREVO_API_KEY를 확인.")
    if not BREVO_FROM_EMAIL:
        raise HTTPException(status_code=500, detail="Brevo 발신 이메일이 설정되지 않았습니다.")

    payload = {
        "sender": {
            "name": BREVO_FROM_NAME,
            "email": BREVO_FROM_EMAIL,
        },
        "to": [
            {"email": to_email}
        ],
        "subject": subject,
        "textContent": body,
    }

    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json",
    }

    try:
        res = requests.post(BREVO_API_URL, headers=headers, json=payload, timeout=20)
        if res.status_code not in (200, 201, 202):
            print("BREVO ERROR:", res.status_code, res.text)
            raise HTTPException(
                status_code=500,
                detail=f"Brevo 이메일 발송 실패: {res.status_code} / {res.text}",
            )
    except HTTPException:
        raise
    except Exception as e:
        print("BREVO REQUEST ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"Brevo 이메일 발송 요청 실패: {str(e)}")

def send_password_find_email(to_email: str, code: str):
    subject = "Alpharion AI 비밀번호 찾기 인증번호"
    body = f"""Alpharion AI 비밀번호 찾기 인증번호입니다.

인증번호: {code}

{PASSWORD_FIND_CODE_EXPIRE_MINUTES}분 이내에 화면에 입력해주세요.
"""
    send_mail(to_email, subject, body)


def send_signup_verification_email(to_email: str, code: str):
    subject = "Alpharion AI 회원가입 인증번호"
    body = f"""Alpharion AI 회원가입 인증번호입니다.

인증번호: {code}

{PASSWORD_FIND_CODE_EXPIRE_MINUTES}분 이내에 회원가입 화면에 입력해주세요.
"""
    send_mail(to_email, subject, body)


def get_signup_verification(email: str):
    conn = get_auth_db()
    row = conn.execute(
        "SELECT * FROM signup_email_verifications WHERE lower(email)=lower(?)",
        (email,),
    ).fetchone()
    conn.close()
    return row


def clear_signup_verification(email: str):
    conn = get_auth_db()
    conn.execute("DELETE FROM signup_email_verifications WHERE lower(email)=lower(?)", (email,))
    conn.commit()
    conn.close()


def verify_signup_email_code(email: str, code: str):
    row = get_signup_verification(email)
    if not row:
        raise HTTPException(status_code=400, detail="회원가입 이메일 인증번호를 먼저 요청해주세요.")

    try:
        expires_at = datetime.fromisoformat(row["expires_at"])
    except Exception:
        clear_signup_verification(email)
        raise HTTPException(status_code=400, detail="회원가입 인증번호가 만료되었습니다. 다시 요청해주세요.")

    if utcnow() > expires_at:
        clear_signup_verification(email)
        raise HTTPException(status_code=400, detail="회원가입 인증번호가 만료되었습니다. 다시 요청해주세요.")

    if not verify_secret(code.strip(), row["code_hash"]):
        raise HTTPException(status_code=400, detail="회원가입 인증번호가 올바르지 않습니다.")


def get_user_by_email(email: str):
    conn = get_auth_db()
    user = conn.execute("SELECT * FROM users WHERE lower(email)=lower(?)", (email,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id: int):
    conn = get_auth_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return user


def parse_iso_dt(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def is_standard_active(user):
    if not user:
        return False
    try:
        plan_type = str(user["plan_type"] or "FREE").upper()
    except Exception:
        plan_type = "FREE"

    expire_at = None
    try:
        expire_at = parse_iso_dt(user["plan_expire_at"])
    except Exception:
        expire_at = None

    return plan_type == "STANDARD" and expire_at is not None and expire_at > utcnow()


def public_user(user):
    if not user:
        return None

    active = is_standard_active(user)

    try:
        plan_type = str(user["plan_type"] or "FREE").upper()
    except Exception:
        plan_type = "FREE"

    if plan_type == "STANDARD" and not active:
        plan_type = "FREE"

    ai_count = int(user["ai_analysis_count"] or 0) if "ai_analysis_count" in user.keys() else 0
    free_limit = int(user["free_ai_limit"] or FREE_AI_LIMIT) if "free_ai_limit" in user.keys() else FREE_AI_LIMIT

    return {
        "id": user["id"],
        "email": user["email"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
        "plan_type": plan_type,
        "standard_active": active,
        "plan_started_at": user["plan_started_at"] if "plan_started_at" in user.keys() else None,
        "plan_expire_at": user["plan_expire_at"] if "plan_expire_at" in user.keys() else None,
        "standard_paid_at": user["standard_paid_at"] if "standard_paid_at" in user.keys() else None,
        "ai_analysis_count": ai_count,
        "free_ai_limit": free_limit,
        "payment_required": (not active and ai_count >= free_limit),
    }


def make_captcha_question():
    ops = ["+", "-", "×"]
    op = secrets.choice(ops)

    if op == "+":
        a = secrets.randbelow(18) + 2
        b = secrets.randbelow(18) + 2
        answer = a + b
    elif op == "-":
        a = secrets.randbelow(25) + 10
        b = secrets.randbelow(9) + 1
        answer = a - b
    else:
        a = secrets.randbelow(8) + 2
        b = secrets.randbelow(8) + 2
        answer = a * b

    return f"{a} {op} {b} = ?", str(answer)


def b64url_encode(data: bytes):
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_decode(data: str):
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def sign_payload(payload: dict):
    body = b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), body.encode("utf-8"), "sha256").digest()
    return f"{body}.{b64url_encode(sig)}"


def verify_signed_payload(token: str):
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(JWT_SECRET_KEY.encode("utf-8"), body.encode("utf-8"), "sha256").digest()
        actual = b64url_decode(sig)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("bad signature")
        payload = json.loads(b64url_decode(body).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(utcnow().timestamp()):
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=400, detail="보안질문이 만료되었습니다. 새로고침 후 다시 시도해주세요.")


def verify_captcha(token: str, answer: str, purpose: str):
    payload = verify_signed_payload(token)
    if payload.get("purpose") != purpose:
        raise HTTPException(status_code=400, detail="보안질문이 올바르지 않습니다.")
    if str(payload.get("answer", "")).strip() != str(answer).strip():
        raise HTTPException(status_code=400, detail="보안질문 정답이 올바르지 않습니다.")


def validate_password_pair(password: str, password_confirm: str, field_name: str = "비밀번호"):
    if password != password_confirm:
        raise HTTPException(status_code=400, detail=f"{field_name}와 재확인이 일치하지 않습니다.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail=f"{field_name}는 최소 6자 이상이어야 합니다.")
    if len(password.encode("utf-8")) > 512:
        raise HTTPException(status_code=400, detail=f"{field_name}가 너무 깁니다. 512바이트 이하로 입력해주세요.")


# =========================================================
# Auth API
# =========================================================
@app.get("/api/auth/captcha")
def auth_captcha(purpose: str = Query("signup")):
    purpose = purpose if purpose in {"signup", "reset", "find_password"} else "signup"
    question, answer = make_captcha_question()
    payload = {
        "purpose": purpose,
        "answer": answer,
        "exp": int((utcnow() + timedelta(minutes=CAPTCHA_EXPIRE_MINUTES)).timestamp()),
        "nonce": secrets.token_urlsafe(8),
    }
    return {"ok": True, "question": question, "token": sign_payload(payload)}


@app.post("/api/auth/signup/request-code")
def auth_signup_request_code(req: SignupEmailCodeRequest):
    email = req.email.strip().lower()

    verify_captcha(req.captcha_token, req.captcha_answer, "signup")

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="이미 가입된 이메일입니다.")

    code = make_email_verification_code()
    expires_at = utcnow() + timedelta(minutes=PASSWORD_FIND_CODE_EXPIRE_MINUTES)
    now = utcnow().isoformat()

    conn = get_auth_db()
    conn.execute(
        """
        INSERT INTO signup_email_verifications (email, code_hash, expires_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            code_hash=excluded.code_hash,
            expires_at=excluded.expires_at,
            updated_at=excluded.updated_at
        """,
        (email, hash_secret(code), expires_at.isoformat(), now, now),
    )
    conn.commit()
    conn.close()

    send_signup_verification_email(email, code)

    return {"ok": True, "message": "회원가입 인증번호를 이메일로 발송했습니다."}


@app.post("/api/auth/signup")
def auth_signup(req: SignupRequest):
    email = req.email.strip().lower()
    password = req.password or ""

    if not req.terms_accepted:
        raise HTTPException(status_code=400, detail="이용약관에 동의해야 회원가입할 수 있습니다.")

    verify_captcha(req.captcha_token, req.captcha_answer, "signup")
    validate_password_pair(password, req.password_confirm, "비밀번호")

    if not req.signup_verification_code.strip():
        raise HTTPException(status_code=400, detail="회원가입 이메일 인증번호를 입력해주세요.")

    verify_signup_email_code(email, req.signup_verification_code)

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="이미 가입된 이메일입니다.")

    recovery_code = make_recovery_code()
    now = utcnow().isoformat()

    conn = get_auth_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (email, password_hash, password_plain, recovery_code_hash, terms_accepted, created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, ?, ?)
        """,
        (email, hash_secret(password), password, hash_secret(recovery_code), now, now),
    )
    conn.commit()
    user_id = cur.lastrowid
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()

    clear_signup_verification(email)

    access_token = create_access_token(user_id, email)

    return {
        "ok": True,
        "message": "회원가입이 완료되었습니다.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": public_user(user),
        "recovery_code": recovery_code,
    }


@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    email = req.email.strip().lower()
    user = get_user_by_email(email)

    if not user or not verify_secret(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")

    token = create_access_token(user["id"], user["email"])
    return {"ok": True, "access_token": token, "token_type": "bearer", "user": public_user(user)}


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="로그인 토큰이 올바르지 않습니다.")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    return user




def verify_admin_secret(x_admin_secret: Optional[str] = Header(None)):
    if not ADMIN_SECRET_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_SECRET_KEY가 Render 환경변수에 설정되지 않았습니다.")
    if not x_admin_secret or not hmac.compare_digest(str(x_admin_secret), str(ADMIN_SECRET_KEY)):
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")
    return True


def row_get(row, key, default=None):
    try:
        if key in row.keys():
            return row[key]
    except Exception:
        pass
    return default


def admin_public_user(row):
    active = is_standard_active(row)
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "terms_accepted": bool(row["terms_accepted"]),
        "plan_type": "STANDARD" if active else str(row["plan_type"] or "FREE").upper(),
        "standard_active": active,
        "plan_started_at": row["plan_started_at"],
        "plan_expire_at": row["plan_expire_at"],
        "standard_paid_at": row["standard_paid_at"],
        "ai_analysis_count": int(row["ai_analysis_count"] or 0),
        "free_ai_limit": int(row["free_ai_limit"] or FREE_AI_LIMIT),
        "last_order_id": row_get(row, "last_order_id"),
        "last_payment_status": row_get(row, "last_payment_status"),
        "last_payment_amount": row_get(row, "last_payment_amount"),
        "last_paid_at": row_get(row, "last_paid_at"),
    }

@app.get("/api/auth/me")
def auth_me(user=Depends(get_current_user)):
    return {"ok": True, "user": public_user(user)}


@app.delete("/api/auth/account")
def auth_delete_account(user=Depends(get_current_user)):
    conn = get_auth_db()
    conn.execute("DELETE FROM users WHERE id=?", (user["id"],))
    conn.commit()
    conn.close()
    return {"ok": True, "message": "계정이 삭제되었습니다."}


@app.post("/api/auth/find-password/request")
def auth_find_password_request(req: PasswordFindRequest):
    email = req.email.strip().lower()

    verify_captcha(req.captcha_token, req.captcha_answer, "find_password")

    user = get_user_by_email(email)
    if not user:
        # 계정 존재 여부를 직접 노출하지 않음
        return {"ok": True, "message": "가입된 이메일이면 인증번호가 발송됩니다."}

    code = make_email_verification_code()
    expires_at = utcnow() + timedelta(minutes=PASSWORD_FIND_CODE_EXPIRE_MINUTES)

    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET find_code_hash=?, find_code_expires_at=?, updated_at=?
        WHERE id=?
        """,
        (hash_secret(code), expires_at.isoformat(), utcnow().isoformat(), user["id"]),
    )
    conn.commit()
    conn.close()

    send_password_find_email(email, code)

    return {"ok": True, "message": "인증번호를 이메일로 발송했습니다."}


@app.post("/api/auth/find-password/confirm")
def auth_find_password_confirm(req: PasswordFindConfirmRequest):
    email = req.email.strip().lower()
    code = req.verification_code.strip()

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=400, detail="이메일 또는 인증번호가 올바르지 않습니다.")

    if not user["find_code_hash"] or not user["find_code_expires_at"]:
        raise HTTPException(status_code=400, detail="인증번호를 먼저 요청해주세요.")

    try:
        expires_at = datetime.fromisoformat(user["find_code_expires_at"])
    except Exception:
        raise HTTPException(status_code=400, detail="인증번호가 만료되었습니다. 다시 요청해주세요.")

    if utcnow() > expires_at:
        raise HTTPException(status_code=400, detail="인증번호가 만료되었습니다. 다시 요청해주세요.")

    if not verify_secret(code, user["find_code_hash"]):
        raise HTTPException(status_code=400, detail="이메일 또는 인증번호가 올바르지 않습니다.")

    password_plain = user["password_plain"] or ""
    if not password_plain:
        raise HTTPException(status_code=400, detail="기존 계정은 비밀번호 표시를 사용할 수 없습니다. 비밀번호 재설정을 이용해주세요.")

    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET find_code_hash=NULL, find_code_expires_at=NULL, updated_at=?
        WHERE id=?
        """,
        (utcnow().isoformat(), user["id"]),
    )
    conn.commit()
    conn.close()

    return {"ok": True, "message": "인증이 완료되었습니다.", "password": password_plain}


@app.post("/api/auth/reset-password")
def auth_reset_password(req: PasswordResetConfirmRequest):
    email = req.email.strip().lower()
    verify_captcha(req.captcha_token, req.captcha_answer, "reset")
    validate_password_pair(req.new_password, req.new_password_confirm, "새 비밀번호")

    user = get_user_by_email(email)
    if not user:
        # 계정 존재 여부를 과도하게 노출하지 않기 위해 일반 메시지로 처리
        raise HTTPException(status_code=400, detail="이메일 또는 복구코드가 올바르지 않습니다.")

    if not verify_secret(req.recovery_code.strip(), user["recovery_code_hash"]):
        raise HTTPException(status_code=400, detail="이메일 또는 복구코드가 올바르지 않습니다.")

    # 재설정 후 복구코드도 새로 발급하는 것이 더 안전하지만,
    # 이메일 발송이 없으므로 사용자가 새 코드를 받지 못합니다.
    # 따라서 현재 복구코드는 유지하고 비밀번호만 변경합니다.
    conn = get_auth_db()
    conn.execute(
        "UPDATE users SET password_hash=?, password_plain=?, updated_at=? WHERE id=?",
        (hash_secret(req.new_password), req.new_password, utcnow().isoformat(), user["id"]),
    )
    conn.commit()
    conn.close()

    return {"ok": True, "message": "비밀번호가 변경되었습니다. 새 비밀번호로 로그인해주세요."}




# =========================================================
# Admin API
# =========================================================
@app.get("/api/admin/users/count")
def admin_users_count(admin_ok=Depends(verify_admin_secret)):
    conn = get_auth_db()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
    conn.close()
    return {"ok": True, "count": int(total)}


@app.get("/api/admin/users")
def admin_users(
    q: str = Query(""),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin_ok=Depends(verify_admin_secret),
):
    q = (q or "").strip().lower()
    conn = get_auth_db()

    if q:
        like = f"%{q}%"
        total = conn.execute(
            "SELECT COUNT(*) AS cnt FROM users WHERE lower(email) LIKE ?",
            (like,),
        ).fetchone()["cnt"]
        rows = conn.execute(
            """
            SELECT u.id, u.email, u.terms_accepted, u.created_at, u.updated_at,
                   u.plan_type, u.plan_started_at, u.plan_expire_at, u.standard_paid_at,
                   u.ai_analysis_count, u.free_ai_limit,
                   (SELECT po.order_id FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_order_id,
                   (SELECT po.status FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_payment_status,
                   (SELECT po.amount FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_payment_amount,
                   (SELECT po.updated_at FROM payment_orders po WHERE po.user_id=u.id AND po.status='PAID' ORDER BY po.id DESC LIMIT 1) AS last_paid_at
            FROM users u
            WHERE lower(u.email) LIKE ?
            ORDER BY u.id DESC
            LIMIT ? OFFSET ?
            """,
            (like, limit, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
        rows = conn.execute(
            """
            SELECT u.id, u.email, u.terms_accepted, u.created_at, u.updated_at,
                   u.plan_type, u.plan_started_at, u.plan_expire_at, u.standard_paid_at,
                   u.ai_analysis_count, u.free_ai_limit,
                   (SELECT po.order_id FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_order_id,
                   (SELECT po.status FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_payment_status,
                   (SELECT po.amount FROM payment_orders po WHERE po.user_id=u.id ORDER BY po.id DESC LIMIT 1) AS last_payment_amount,
                   (SELECT po.updated_at FROM payment_orders po WHERE po.user_id=u.id AND po.status='PAID' ORDER BY po.id DESC LIMIT 1) AS last_paid_at
            FROM users u
            ORDER BY u.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    conn.close()
    return {
        "ok": True,
        "count": int(total),
        "limit": int(limit),
        "offset": int(offset),
        "users": [admin_public_user(row) for row in rows],
    }


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, admin_ok=Depends(verify_admin_secret)):
    conn = get_auth_db()
    user = conn.execute("SELECT id, email FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="해당 회원을 찾을 수 없습니다.")

    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    return {"ok": True, "message": "회원 계정이 삭제되었습니다.", "deleted_user_id": user_id, "deleted_email": user["email"]}


def parse_admin_paid_at(value: Optional[str]):
    if not value:
        return utcnow()
    text = str(value).strip()
    if not text:
        return utcnow()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        raise HTTPException(status_code=400, detail="결제일 형식이 올바르지 않습니다. 예: 2026-05-17T15:30:00")


@app.post("/api/admin/users/{user_id}/standard")
def admin_set_standard_user(user_id: int, req: AdminSetStandardRequest, admin_ok=Depends(verify_admin_secret)):
    days = int(req.days or STANDARD_PLAN_DAYS)
    if days < 1 or days > 3660:
        raise HTTPException(status_code=400, detail="Standard 적용 기간은 1일 이상 3660일 이하로 입력해주세요.")

    paid_at = parse_admin_paid_at(req.paid_at)
    expire_at = paid_at + timedelta(days=days)
    now = utcnow().isoformat()
    manual_order_id = create_order_id(user_id) + "MANUAL"
    goods_name = f"관리자 수동 Standard {days}일 처리"

    conn = get_auth_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="해당 회원을 찾을 수 없습니다.")

    conn.execute(
        """
        UPDATE users
        SET plan_type='STANDARD',
            plan_started_at=?,
            plan_expire_at=?,
            standard_paid_at=?,
            ai_analysis_count=CASE WHEN ? THEN 0 ELSE ai_analysis_count END,
            updated_at=?
        WHERE id=?
        """,
        (paid_at.isoformat(), expire_at.isoformat(), paid_at.isoformat(), 1 if req.reset_free_count else 0, now, user_id),
    )

    conn.execute(
        """
        INSERT INTO payment_orders (order_id, user_id, amount, status, goods_name, raw_prepare, raw_approve, created_at, updated_at)
        VALUES (?, ?, ?, 'PAID', ?, ?, ?, ?, ?)
        """,
        (
            manual_order_id,
            user_id,
            STANDARD_PRICE_KRW,
            goods_name,
            json.dumps({"manual_admin": True, "days": days}, ensure_ascii=False),
            json.dumps({"manual_admin": True, "paid_at": paid_at.isoformat(), "expire_at": expire_at.isoformat()}, ensure_ascii=False),
            paid_at.isoformat(),
            now,
        ),
    )

    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()

    return {"ok": True, "message": "Standard 유료회원으로 변경했습니다.", "user": public_user(updated)}


@app.post("/api/admin/users/{user_id}/free")
def admin_set_free_user(user_id: int, req: AdminSetFreeRequest, admin_ok=Depends(verify_admin_secret)):
    now = utcnow().isoformat()
    conn = get_auth_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="해당 회원을 찾을 수 없습니다.")

    conn.execute(
        """
        UPDATE users
        SET plan_type='FREE',
            plan_started_at=NULL,
            plan_expire_at=NULL,
            standard_paid_at=NULL,
            ai_analysis_count=CASE WHEN ? THEN 0 ELSE ai_analysis_count END,
            updated_at=?
        WHERE id=?
        """,
        (1 if req.reset_free_count else 0, now, user_id),
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return {"ok": True, "message": "FREE 회원으로 변경했습니다.", "user": public_user(updated)}


# =========================================================
# Plan / Payment Helpers
# =========================================================
def create_order_id(user_id: int):
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand = secrets.token_hex(4).upper()
    return f"AMW{stamp}{user_id}{rand}"


def mark_standard_paid(user_id: int, order_id: str = "", tid: str = "", raw_approve: Optional[dict] = None):
    now = utcnow()
    expire_at = now + timedelta(days=STANDARD_PLAN_DAYS)

    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET plan_type='STANDARD',
            plan_started_at=?,
            plan_expire_at=?,
            standard_paid_at=?,
            updated_at=?
        WHERE id=?
        """,
        (now.isoformat(), expire_at.isoformat(), now.isoformat(), now.isoformat(), user_id),
    )

    if order_id:
        conn.execute(
            """
            UPDATE payment_orders
            SET status='PAID',
                tid=?,
                raw_approve=?,
                updated_at=?
            WHERE order_id=?
            """,
            (tid or "", json.dumps(raw_approve or {}, ensure_ascii=False), now.isoformat(), order_id),
        )

    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return user


def consume_ai_analysis_or_raise(user):
    if is_standard_active(user):
        return

    used = int(user["ai_analysis_count"] or 0)
    limit = int(user["free_ai_limit"] or FREE_AI_LIMIT)

    if used >= limit:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FREE_LIMIT_EXCEEDED",
                "message": "AI 종목분석 무료 사용 5회를 모두 사용했습니다.",
                "payment_required": True,
            },
        )

    conn = get_auth_db()
    conn.execute(
        "UPDATE users SET ai_analysis_count=ai_analysis_count+1, updated_at=? WHERE id=?",
        (utcnow().isoformat(), user["id"]),
    )
    conn.commit()
    conn.close()


def nicepay_basic_auth_header():
    raw = f"{NICEPAY_CLIENT_ID}:{NICEPAY_SECRET_KEY}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("utf-8")


@app.post("/api/payments/nicepay/prepare")
def nicepay_prepare(req: NicepayPrepareRequest, user=Depends(get_current_user)):
    fresh_user = get_user_by_id(user["id"])
    if is_standard_active(fresh_user):
        return {
            "ok": True,
            "already_paid": True,
            "message": "Standard 이용기간이 아직 남아 있습니다.",
            "user": public_user(fresh_user),
        }

    if not NICEPAY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="NICEPAY_CLIENT_ID가 Render 환경변수에 설정되지 않았습니다.")

    order_id = create_order_id(user["id"])
    amount = int(STANDARD_PRICE_KRW)
    goods_name = "Alpharion Standard 1개월 이용권"
    now = utcnow().isoformat()

    conn = get_auth_db()
    conn.execute(
        """
        INSERT INTO payment_orders (order_id, user_id, amount, status, goods_name, created_at, updated_at)
        VALUES (?, ?, ?, 'READY', ?, ?, ?)
        """,
        (order_id, user["id"], amount, goods_name, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "client_id": NICEPAY_CLIENT_ID,
        "method": "card",
        "order_id": order_id,
        "amount": amount,
        "goods_name": goods_name,
        "return_url": f"{API_PUBLIC_BASE}/api/payments/nicepay/return",
        "buyer_email": user["email"],
        "buyer_name": user["email"].split("@")[0],
        "mall_user_id": str(user["id"]),
    }


def nicepay_pick(params: dict, *names: str, default: str = ""):
    """NICEPAY가 환경에 따라 tid/TID, orderId/order_id처럼 다른 키를 보낼 수 있어 안전하게 읽습니다."""
    if not params:
        return default
    lower_map = {str(k).lower(): v for k, v in params.items()}
    for name in names:
        if name in params and params.get(name) not in (None, ""):
            return params.get(name)
        key = str(name).lower()
        if key in lower_map and lower_map.get(key) not in (None, ""):
            return lower_map.get(key)
    return default


def nicepay_to_int_amount(value):
    text = str(value or "0").replace(",", "").strip()
    try:
        return int(float(text))
    except Exception:
        return 0


@app.get("/api/payments/nicepay/return", response_class=HTMLResponse)
@app.post("/api/payments/nicepay/return", response_class=HTMLResponse)
async def nicepay_return(request: Request):
    if request.method == "POST":
        form = await request.form()
        params = dict(form)
    else:
        params = dict(request.query_params)

    print("NICEPAY RETURN PARAMS:", json.dumps(params, ensure_ascii=False))

    result_code = str(nicepay_pick(params, "authResultCode", "resultCode", "ResultCode")).strip()
    result_msg = str(nicepay_pick(params, "authResultMsg", "resultMsg", "ResultMsg", default="")).strip()
    order_id = str(nicepay_pick(params, "orderId", "order_id", "OrderId", "moid", "Moid", default="")).strip()
    amount = nicepay_to_int_amount(nicepay_pick(params, "amount", "Amount", "amt", "Amt", default="0"))

    # NICEPAY 승인 API는 일반적으로 TID를 path에 넣어 호출합니다.
    # 일부 응답에서는 authToken만 올 수 있어 authToken도 fallback으로 사용합니다.
    tid = str(nicepay_pick(params, "tid", "TID", "Tid", default="")).strip()
    auth_token = str(nicepay_pick(params, "authToken", "auth_token", "AuthToken", default="")).strip()
    approve_key = tid or auth_token

    if result_code and result_code != "0000":
        return HTMLResponse(
            f"""
            <script>
              alert({json.dumps('결제가 완료되지 않았습니다. ' + (result_msg or ''), ensure_ascii=False)});
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    if not order_id or not amount:
        return HTMLResponse(
            f"""
            <script>
              alert("결제 주문번호 또는 금액 정보가 부족합니다.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    if not approve_key:
        conn = get_auth_db()
        conn.execute(
            "UPDATE payment_orders SET status='FAILED', raw_approve=?, updated_at=? WHERE order_id=?",
            (json.dumps({"error": "MISSING_TID_OR_AUTHTOKEN", "return_params": params}, ensure_ascii=False), utcnow().isoformat(), order_id),
        )
        conn.commit()
        conn.close()
        return HTMLResponse(
            f"""
            <script>
              alert("결제 승인번호 TID를 받지 못했습니다. NICEPAY 설정을 확인해주세요.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    conn = get_auth_db()
    order = conn.execute("SELECT * FROM payment_orders WHERE order_id=?", (order_id,)).fetchone()
    conn.close()

    if not order:
        return HTMLResponse(
            f"""
            <script>
              alert("결제 주문 정보를 찾을 수 없습니다.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    if int(order["amount"]) != amount:
        return HTMLResponse(
            f"""
            <script>
              alert("결제 금액이 일치하지 않습니다.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    if not NICEPAY_SECRET_KEY:
        return HTMLResponse(
            f"""
            <script>
              alert("결제 승인 설정이 완료되지 않았습니다.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )

    approve_payload = {"amount": amount}
    approve_headers = {
        "Content-Type": "application/json",
        "Authorization": nicepay_basic_auth_header(),
    }

    try:
        # 핵심 수정: authToken이 아니라 TID 우선으로 승인 API 호출
        approve_res = requests.post(
            f"{NICEPAY_APPROVE_URL}/{approve_key}",
            headers=approve_headers,
            json=approve_payload,
            timeout=25,
        )
        try:
            approve_data = approve_res.json() if approve_res.text else {}
        except Exception:
            approve_data = {"raw_text": approve_res.text}

        print("NICEPAY APPROVE RESPONSE:", approve_res.status_code, json.dumps(approve_data, ensure_ascii=False))

        approve_result_code = str(approve_data.get("resultCode") or approve_data.get("ResultCode") or "")
        if approve_res.status_code not in (200, 201) or approve_result_code != "0000":
            msg = approve_data.get("resultMsg") or approve_data.get("ResultMsg") or approve_data.get("message") or "결제 승인에 실패했습니다."
            conn = get_auth_db()
            conn.execute(
                """
                UPDATE payment_orders
                SET status='FAILED', auth_token=?, tid=?, raw_approve=?, updated_at=?
                WHERE order_id=?
                """,
                (auth_token or approve_key, tid or approve_key, json.dumps(approve_data, ensure_ascii=False), utcnow().isoformat(), order_id),
            )
            conn.commit()
            conn.close()
            return HTMLResponse(
                f"""
                <script>
                  alert({json.dumps(str(msg), ensure_ascii=False)});
                  location.href="{FRONTEND_BASE_URL}/#payment";
                </script>
                """
            )

        approved_tid = approve_data.get("tid") or approve_data.get("TID") or tid or approve_key
        mark_standard_paid(int(order["user_id"]), order_id=order_id, tid=approved_tid, raw_approve=approve_data)

        return HTMLResponse(
            f"""
            <script>
              alert("Standard 결제가 완료되었습니다. 계정정보에서 이용기간을 확인할 수 있습니다.");
              location.href="{FRONTEND_BASE_URL}/";
            </script>
            """
        )

    except Exception as e:
        print("NICEPAY APPROVE ERROR:", repr(e))
        conn = get_auth_db()
        conn.execute(
            "UPDATE payment_orders SET status='ERROR', auth_token=?, tid=?, raw_approve=?, updated_at=? WHERE order_id=?",
            (auth_token or approve_key, tid or approve_key, json.dumps({"error": repr(e), "return_params": params}, ensure_ascii=False), utcnow().isoformat(), order_id),
        )
        conn.commit()
        conn.close()
        return HTMLResponse(
            f"""
            <script>
              alert("결제 승인 처리 중 오류가 발생했습니다. 관리자에게 문의해주세요.");
              location.href="{FRONTEND_BASE_URL}/#payment";
            </script>
            """
        )


# =========================================================
# Standard Stock Screener API
# =========================================================
SCREENER_PATTERN_LABELS = {
    "uptrend_pullback": "상승추세 조정 패턴",
    "golden_cross": "골든크로스",
    "breakout": "신고가 돌파",
    "higher_low": "저점 상승",
    "bottom_turn": "둥근 바닥",
    "support_rebound": "지지선 반등",
    "downtrend": "하락추세 이탈",
    "v_reversal": "V자 반등",
}

SCREENER_FINANCIAL_LABELS = {
    "eps_growth": "주당 순이익 증가",
    "roe_high": "자기자본이익률 상위",
    "debt_low": "부채비율 감소",
    "revenue_growth": "매출 성장",
    "profit_margin_high": "순이익률 상위",
    "value_stock": "밸류에이션 저평가",
    "cashflow_high": "현금흐름 양호",
    "dividend_high": "배당률 상위",
    "low_pbr": "저 PBR",
    "low_per": "저 PER",
    "sales_growth": "매출액 증가율 상위",
    "income_growth": "순이익 증가율 상위",
    "equity_growth": "자기자본 증가율 상위",
    "turnaround": "흑자 전환",
}

US_SCREENER_UNIVERSE = [
    ("Apple", "AAPL"), ("Microsoft", "MSFT"), ("NVIDIA", "NVDA"), ("Tesla", "TSLA"),
    ("Amazon", "AMZN"), ("Alphabet A", "GOOGL"), ("Meta Platforms", "META"), ("AMD", "AMD"),
    ("Broadcom", "AVGO"), ("Palantir", "PLTR"), ("Super Micro Computer", "SMCI"), ("TSMC", "TSM"),
    ("JPMorgan", "JPM"), ("Berkshire Hathaway", "BRK-B"), ("Visa", "V"), ("Eli Lilly", "LLY"),
    ("Netflix", "NFLX"), ("Costco", "COST"), ("Exxon Mobil", "XOM"), ("Chevron", "CVX"),
    ("SPDR S&P 500 ETF", "SPY"), ("Invesco QQQ", "QQQ"), ("Technology ETF", "XLK"), ("Semiconductor ETF", "SOXX"),
]

SCREENER_SECTOR_UNIVERSE = {
    "AI": {
        "KR": [("삼성전자", "005930.KS"), ("SK하이닉스", "000660.KS"), ("NAVER", "035420.KS"), ("카카오", "035720.KS")],
        "US": [("NVIDIA", "NVDA"), ("Microsoft", "MSFT"), ("Alphabet A", "GOOGL"), ("Meta Platforms", "META"), ("Palantir", "PLTR"), ("AMD", "AMD"), ("Broadcom", "AVGO"), ("Super Micro Computer", "SMCI")],
    },
    "SEMICONDUCTOR": {
        "KR": [("삼성전자", "005930.KS"), ("SK하이닉스", "000660.KS"), ("DB하이텍", "000990.KS"), ("한미반도체", "042700.KS"), ("이오테크닉스", "039030.KQ")],
        "US": [("NVIDIA", "NVDA"), ("AMD", "AMD"), ("Broadcom", "AVGO"), ("TSMC", "TSM"), ("Intel", "INTC"), ("Qualcomm", "QCOM"), ("Micron", "MU"), ("Semiconductor ETF", "SOXX")],
    },
    "BATTERY": {
        "KR": [("LG에너지솔루션", "373220.KS"), ("삼성SDI", "006400.KS"), ("LG화학", "051910.KS"), ("포스코퓨처엠", "003670.KS"), ("에코프로", "086520.KQ")],
        "US": [("Tesla", "TSLA"), ("Albemarle", "ALB"), ("QuantumScape", "QS"), ("Global X Lithium ETF", "LIT")],
    },
    "BIO": {
        "KR": [("셀트리온", "068270.KS"), ("삼성바이오로직스", "207940.KS"), ("알테오젠", "196170.KQ"), ("HLB", "028300.KQ")],
        "US": [("Eli Lilly", "LLY"), ("Johnson & Johnson", "JNJ"), ("Pfizer", "PFE"), ("Merck", "MRK"), ("Moderna", "MRNA")],
    },
    "DEFENSE": {
        "KR": [("한화에어로스페이스", "012450.KS"), ("현대로템", "064350.KS"), ("한화오션", "042660.KS"), ("LIG넥스원", "079550.KS"), ("한국항공우주", "047810.KS")],
        "US": [("Lockheed Martin", "LMT"), ("Northrop Grumman", "NOC"), ("RTX", "RTX"), ("General Dynamics", "GD")],
    },
    "NUCLEAR": {
        "KR": [("두산에너빌리티", "034020.KS"), ("한전기술", "052690.KS"), ("한전KPS", "051600.KS"), ("한국전력", "015760.KS")],
        "US": [("Cameco", "CCJ"), ("Constellation Energy", "CEG"), ("Uranium Energy", "UEC"), ("NuScale Power", "SMR")],
    },
    "ROBOT": {
        "KR": [("레인보우로보틱스", "277810.KQ"), ("로보티즈", "108490.KQ"), ("유진로봇", "056080.KQ"), ("두산로보틱스", "454910.KS")],
        "US": [("Tesla", "TSLA"), ("Intuitive Surgical", "ISRG"), ("Rockwell Automation", "ROK"), ("Teradyne", "TER")],
    },
    "SHIPBUILDING": {
        "KR": [("HD현대중공업", "329180.KS"), ("HD한국조선해양", "009540.KS"), ("한화오션", "042660.KS"), ("삼성중공업", "010140.KS"), ("팬오션", "028670.KS"), ("HMM", "011200.KS"), ("대한해운", "005880.KS")],
        "US": [("ZIM Integrated Shipping", "ZIM"), ("Star Bulk Carriers", "SBLK"), ("Danaos", "DAC"), ("Matson", "MATX")],
    },
    "AUTO": {
        "KR": [("현대차", "005380.KS"), ("기아", "000270.KS"), ("현대모비스", "012330.KS"), ("HL만도", "204320.KS")],
        "US": [("Tesla", "TSLA"), ("Ford", "F"), ("General Motors", "GM"), ("Rivian", "RIVN")],
    },
    "BANK": {
        "KR": [("KB금융", "105560.KS"), ("신한지주", "055550.KS"), ("하나금융지주", "086790.KS"), ("우리금융지주", "316140.KS"), ("기업은행", "024110.KS")],
        "US": [("JPMorgan", "JPM"), ("Bank of America", "BAC"), ("Wells Fargo", "WFC"), ("Goldman Sachs", "GS"), ("Morgan Stanley", "MS")],
    },
    "INTERNET": {
        "KR": [("NAVER", "035420.KS"), ("카카오", "035720.KS"), ("엔씨소프트", "036570.KS"), ("크래프톤", "259960.KS")],
        "US": [("Alphabet A", "GOOGL"), ("Meta Platforms", "META"), ("Amazon", "AMZN"), ("Netflix", "NFLX")],
    },
    "ETF": {
        "KR": [("KODEX 200", "069500.KS"), ("TIGER 200", "102110.KS"), ("KODEX 코스닥150", "229200.KS")],
        "US": [("SPDR S&P 500 ETF", "SPY"), ("Invesco QQQ", "QQQ"), ("Technology ETF", "XLK"), ("Semiconductor ETF", "SOXX")],
    },
}

SCREENER_SECTOR_ALIASES = {
    "ALL": "ALL", "": "ALL",
    "AI": "AI", "인공지능": "AI",
    "반도체": "SEMICONDUCTOR", "SEMICONDUCTOR": "SEMICONDUCTOR", "HBM": "SEMICONDUCTOR",
    "2차전지": "BATTERY", "배터리": "BATTERY", "BATTERY": "BATTERY",
    "바이오": "BIO", "헬스케어": "BIO", "BIO": "BIO",
    "방산": "DEFENSE", "우주항공": "DEFENSE", "DEFENSE": "DEFENSE",
    "원전": "NUCLEAR", "에너지": "NUCLEAR", "NUCLEAR": "NUCLEAR",
    "로봇": "ROBOT", "ROBOT": "ROBOT",
    "조선": "SHIPBUILDING", "해운": "SHIPBUILDING", "SHIPPING": "SHIPBUILDING", "SHIPBUILDING": "SHIPBUILDING",
    "자동차": "AUTO", "모빌리티": "AUTO", "AUTO": "AUTO",
    "은행": "BANK", "금융": "BANK", "BANK": "BANK",
    "인터넷": "INTERNET", "플랫폼": "INTERNET", "INTERNET": "INTERNET",
    "ETF": "ETF", "지수": "ETF",
}


def normalize_screener_sector(keyword: str):
    text = str(keyword or "ALL").strip()
    return SCREENER_SECTOR_ALIASES.get(text, SCREENER_SECTOR_ALIASES.get(text.upper(), "ALL"))


def build_screener_universe(market: str, keyword: str, limit: int):
    market = (market or "ALL").upper()
    sector = normalize_screener_sector(keyword)
    result = []
    seen = set()
    max_limit = max(1, min(int(limit or 40), 200))

    def add_item(name, symbol, item_market):
        if not symbol or symbol in seen:
            return
        seen.add(symbol)
        result.append({"name": name or symbol, "symbol": symbol, "market": item_market})

    # 사용자가 섹터를 선택한 경우: 해당 섹터의 사전 정의 유니버스만 검색합니다.
    if sector != "ALL" and sector in SCREENER_SECTOR_UNIVERSE:
        sector_items = SCREENER_SECTOR_UNIVERSE[sector]
        if market in {"KR", "ALL"}:
            for name, symbol in sector_items.get("KR", []):
                add_item(name, symbol, "Korea")
        if market in {"US", "ALL"}:
            for name, symbol in sector_items.get("US", []):
                add_item(name, symbol, "US")
        return result[:max_limit]

    # 전체 섹터인 경우: 기존처럼 한국 대표 종목 + KRX 일부 + 미국 주요주를 대상으로 검색합니다.
    if market in {"KR", "ALL"}:
        for name, symbol in KOREAN_NAME_MAP.items():
            add_item(name, symbol, "Korea")
        for item in get_krx_stocks():
            add_item(item.get("name") or item.get("symbol"), item.get("symbol"), item.get("market") or "Korea")
            if len(result) >= max_limit and market == "KR":
                break

    if market in {"US", "ALL"}:
        for name, symbol in US_SCREENER_UNIVERSE:
            add_item(name, symbol, "US")

    return result[:max_limit]


def safe_float_value(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def detect_chart_patterns(close_prices, ma5, ma20, ma60, high_prices, low_prices):
    patterns = {}
    if len(close_prices) < 60:
        return patterns
    last = float(close_prices[-1])
    prev = float(close_prices[-2]) if len(close_prices) >= 2 else last
    recent_high = max([float(x) for x in high_prices[-60:] if not is_bad_number(x)] or [last])
    recent_low = min([float(x) for x in low_prices[-60:] if not is_bad_number(x)] or [last])
    ma5_now = safe_float_value(ma5[-1])
    ma20_now = safe_float_value(ma20[-1])
    ma60_now = safe_float_value(ma60[-1])
    ma5_prev = safe_float_value(ma5[-2]) if len(ma5) >= 2 else None
    ma20_prev = safe_float_value(ma20[-2]) if len(ma20) >= 2 else None

    patterns["golden_cross"] = bool(ma5_prev is not None and ma20_prev is not None and ma5_now is not None and ma20_now is not None and ma5_prev <= ma20_prev and ma5_now > ma20_now)
    patterns["breakout"] = bool(last >= recent_high * 0.985 and last > prev)
    patterns["higher_low"] = bool(len(low_prices) >= 40 and min(low_prices[-20:]) > min(low_prices[-40:-20]) * 1.02)
    patterns["bottom_turn"] = bool(last > min(close_prices[-40:]) * 1.12 and ma20_now is not None and last > ma20_now)
    patterns["support_rebound"] = bool(recent_low > 0 and last > recent_low * 1.08 and prev <= last)
    patterns["downtrend"] = bool(ma20_now is not None and ma60_now is not None and last > ma20_now and ma20_now < ma60_now)
    patterns["v_reversal"] = bool(len(close_prices) >= 25 and min(close_prices[-20:-5]) < close_prices[-25] * 0.9 and last > close_prices[-5] * 1.05)
    patterns["uptrend_pullback"] = bool(ma20_now is not None and ma60_now is not None and ma20_now > ma60_now and last >= ma20_now * 0.97 and last <= ma20_now * 1.08)
    return patterns


def evaluate_financial_filters(info: dict):
    pe = safe_float_value(info.get("trailingPE"))
    pb = safe_float_value(info.get("priceToBook"))
    roe = safe_float_value(info.get("returnOnEquity"))
    roa = safe_float_value(info.get("returnOnAssets"))
    debt = safe_float_value(info.get("debtToEquity"))
    margin = safe_float_value(info.get("profitMargins"))
    revenue_growth = safe_float_value(info.get("revenueGrowth"))
    earnings_growth = safe_float_value(info.get("earningsGrowth"))
    dividend = safe_float_value(info.get("dividendYield"))
    operating_cashflow = safe_float_value(info.get("operatingCashflow"))
    free_cashflow = safe_float_value(info.get("freeCashflow"))

    checks = {
        "low_per": pe is not None and 0 < pe <= 18,
        "low_pbr": pb is not None and 0 < pb <= 1.3,
        "roe_high": roe is not None and roe >= 0.10,
        "debt_low": debt is not None and debt <= 120,
        "profit_margin_high": margin is not None and margin >= 0.08,
        "revenue_growth": revenue_growth is not None and revenue_growth >= 0.05,
        "sales_growth": revenue_growth is not None and revenue_growth >= 0.08,
        "income_growth": earnings_growth is not None and earnings_growth >= 0.05,
        "eps_growth": earnings_growth is not None and earnings_growth >= 0.05,
        "dividend_high": dividend is not None and dividend >= 0.02,
        "cashflow_high": (operating_cashflow is not None and operating_cashflow > 0) or (free_cashflow is not None and free_cashflow > 0),
        "value_stock": (pe is not None and 0 < pe <= 15) or (pb is not None and 0 < pb <= 1.0),
        "equity_growth": roa is not None and roa >= 0.04,
        "turnaround": earnings_growth is not None and earnings_growth >= 0.20,
    }
    return checks


def normalize_screener_operator(value: str):
    value = str(value or "AND").upper().strip()
    return "OR" if value == "OR" else "AND"


def normalize_screener_filter_groups(req: ScreenerRequest):
    groups = []

    # 새 방식: 사용자가 직접 만든 그룹 구조
    for idx, group in enumerate(req.groups or []):
        filters = []
        for f in group.filters or []:
            f_type = str(f.type or "").strip().lower()
            key = str(f.key or "").strip()
            if f_type == "pattern" and key in SCREENER_PATTERN_LABELS:
                filters.append({"type": "pattern", "key": key})
            elif f_type == "financial" and key in SCREENER_FINANCIAL_LABELS:
                filters.append({"type": "financial", "key": key})
        if filters:
            groups.append({
                "id": group.id or f"G{idx + 1}",
                "name": group.name or f"그룹 {idx + 1}",
                "operator": normalize_screener_operator(group.operator),
                "filters": filters,
            })

    # 기존 방식 호환: patterns/financials만 넘어오는 경우 전체 AND 그룹으로 처리
    if not groups:
        legacy_filters = []
        for p in (req.patterns or []):
            if p in SCREENER_PATTERN_LABELS:
                legacy_filters.append({"type": "pattern", "key": p})
        for f in (req.financials or []):
            if f in SCREENER_FINANCIAL_LABELS:
                legacy_filters.append({"type": "financial", "key": f})
        if legacy_filters:
            groups.append({"id": "G1", "name": "기본 그룹", "operator": "AND", "filters": legacy_filters})

    return groups


def get_screener_label(filter_type: str, key: str):
    if filter_type == "pattern":
        return SCREENER_PATTERN_LABELS.get(key, key)
    if filter_type == "financial":
        return SCREENER_FINANCIAL_LABELS.get(key, key)
    return key


def evaluate_screener_groups(pattern_checks: dict, financial_checks: dict, groups: list, group_operator: str = "AND"):
    group_operator = normalize_screener_operator(group_operator)
    group_results = []
    matched_keys = []
    matched_labels = []

    for group in groups:
        values = []
        true_count = 0
        for f in group.get("filters", []):
            f_type = f.get("type")
            key = f.get("key")
            passed = bool(pattern_checks.get(key)) if f_type == "pattern" else bool(financial_checks.get(key))
            values.append(passed)
            if passed:
                true_count += 1
                matched_keys.append(key)
                matched_labels.append(get_screener_label(f_type, key))

        op = normalize_screener_operator(group.get("operator", "AND"))
        group_passed = any(values) if op == "OR" else all(values)
        group_results.append({
            "id": group.get("id"),
            "name": group.get("name"),
            "operator": op,
            "passed": bool(group_passed),
            "total": len(values),
            "matched": true_count,
        })

    if not group_results:
        return False, [], [], []

    final_ok = any(g["passed"] for g in group_results) if group_operator == "OR" else all(g["passed"] for g in group_results)
    return bool(final_ok), matched_keys, matched_labels, group_results


def analyze_screener_symbol(item: dict, filter_groups: list, group_operator: str = "AND"):
    symbol = item["symbol"]
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d").dropna()
    if hist.empty or len(hist) < 60:
        return None

    close_prices = hist["Close"].astype(float).tolist()
    high_prices = hist["High"].astype(float).tolist()
    low_prices = hist["Low"].astype(float).tolist()
    ma5 = moving_average(close_prices, 5)
    ma20 = moving_average(close_prices, 20)
    ma60 = moving_average(close_prices, 60)
    rsi = calculate_rsi(close_prices)
    period_return = ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100

    pattern_checks = detect_chart_patterns(close_prices, ma5, ma20, ma60, high_prices, low_prices)

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    financial_checks = evaluate_financial_filters(info)

    final_ok, matched_keys, matched_labels, group_results = evaluate_screener_groups(
        pattern_checks, financial_checks, filter_groups, group_operator
    )

    if not final_ok:
        return None

    currency = "KRW" if symbol.endswith(".KS") or symbol.endswith(".KQ") else "USD"
    name = item.get("name") or info.get("shortName") or info.get("longName") or symbol
    logic_text = "그룹 중 하나 이상 통과" if normalize_screener_operator(group_operator) == "OR" else "모든 그룹 통과"
    summary = f"조건검색을 통과했습니다. {logic_text}, 매칭 {len(matched_labels)}개, 1년 수익률 {round(period_return, 2)}%, RSI {round(float(rsi), 1)}"
    return {
        "symbol": symbol,
        "name": name,
        "market": item.get("market") or ("Korea" if currency == "KRW" else "US"),
        "currency": currency,
        "last_price": round(float(close_prices[-1]), 2),
        "period_return": round(float(period_return), 2),
        "rsi": round(float(rsi), 1),
        "matched_labels": matched_labels,
        "group_results": group_results,
        "summary": summary,
    }


@app.post("/api/screener")
def stock_screener(req: ScreenerRequest, user=Depends(get_current_user)):
    fresh_user = get_user_by_id(user["id"])
    if not is_standard_active(fresh_user):
        raise HTTPException(
            status_code=403,
            detail={"code": "STANDARD_REQUIRED", "message": "Stock Screener는 Standard 회원 전용 기능입니다."},
        )

    filter_groups = normalize_screener_filter_groups(req)
    if not filter_groups:
        raise HTTPException(status_code=400, detail="검색할 필터를 1개 이상 선택해주세요.")

    group_operator = normalize_screener_operator(req.group_operator)

    universe = build_screener_universe(req.market, req.keyword, req.limit)
    results = []
    errors = 0
    for item in universe:
        try:
            row = analyze_screener_symbol(item, filter_groups, group_operator)
            if row:
                results.append(row)
        except Exception:
            errors += 1
            continue
        if len(results) >= 30:
            break

    results = sorted(results, key=lambda x: (len(x.get("matched_labels", [])), x.get("period_return", 0)), reverse=True)
    return {
        "ok": True,
        "count": len(results),
        "checked": len(universe),
        "errors": errors,
        "group_operator": group_operator,
        "groups": filter_groups,
        "results": results,
    }

# =========================================================
# Stock / Market API
# =========================================================
KOREAN_NAME_MAP = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "LG에너지솔루션": "373220.KS",
    "LG화학": "051910.KS",
    "삼성SDI": "006400.KS",
    "POSCO홀딩스": "005490.KS",
    "셀트리온": "068270.KS",
    "삼성바이오로직스": "207940.KS",
    "두산에너빌리티": "034020.KS",
    "한화오션": "042660.KS",
    "현대로템": "064350.KS",
    "한화에어로스페이스": "012450.KS",
    "산일전기": "062040.KS",
    "팬오션": "028670.KS",
}

KRX_CACHE = None


@app.get("/")
def root():
    return {"service": "Alpharion Market Watch", "company": "CodeGeneva Inc.", "status": "running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/news")
def get_market_issue_news():
    queries = [
        "stock market today",
        "market movers",
        "nasdaq stocks today",
        "AI stocks semiconductor market",
        "Federal Reserve interest rates stocks",
        "oil prices inflation stocks",
        "Korea stock market",
        "global markets today",
        "earnings stock market",
        "ETF market trends",
    ]

    collected = []
    seen_titles = set()

    for query in queries:
        for item in yahoo_market_news_search(query):
            title = item.get("title", "").strip()
            if not title:
                continue

            title_key = normalize_text(title)
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            collected.append(
                {
                    "category": classify_market_news(title),
                    "symbol": item.get("related", "Market"),
                    "title": title,
                    "publisher": item.get("publisher", "Market News"),
                    "link": item.get("link", ""),
                    "date": item.get("date", ""),
                    "summary": make_news_summary(title),
                    "importance": score_market_news(title),
                }
            )

    collected = sorted(collected, key=lambda x: x.get("importance", 0), reverse=True)
    if not collected:
        collected = fallback_market_news()

    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Yahoo Finance market issue search",
        "items": collected[:18],
    }


@app.get("/api/search")
def search_stock(q: str = Query("")):
    q = q.strip()
    if not q:
        return []

    results = []
    seen = set()

    for name, symbol in KOREAN_NAME_MAP.items():
        if normalize_text(q) in normalize_text(name) or normalize_text(q) in normalize_text(symbol):
            item = {"name": name, "symbol": symbol, "market": "Korea", "type": "EQUITY"}
            results.append(item)
            seen.add(symbol)

    for item in search_krx_by_name(q):
        symbol = item["symbol"]
        if symbol not in seen:
            results.append(item)
            seen.add(symbol)

    if q.isdigit() and len(q) == 6:
        for suffix, market in [(".KS", "Korea"), (".KQ", "Korea KOSDAQ")]:
            symbol = q + suffix
            if symbol not in seen:
                results.append({"name": q, "symbol": symbol, "market": market, "type": "EQUITY"})
                seen.add(symbol)

    for item in yahoo_search(q):
        symbol = item.get("symbol")
        if symbol and symbol not in seen:
            results.append(item)
            seen.add(symbol)

    if not results:
        guessed = normalize_symbol(q)
        results.append({"name": guessed, "symbol": guessed, "market": "Direct Ticker", "type": "UNKNOWN"})

    return results[:30]


@app.get("/api/stock/{symbol}")
def get_stock(symbol: str, period: str = "1y", user=Depends(get_current_user)):
    original_input = symbol
    symbol = normalize_symbol(symbol)
    period = validate_period(period)

    # 무료회원 5회 제한은 서버에서 강제 적용합니다.
    # 브라우저 localStorage 값은 믿지 않고 DB의 최신 회원정보를 다시 읽습니다.
    fresh_user = get_user_by_id(user["id"])
    consume_ai_analysis_or_raise(fresh_user)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")

        if hist.empty and symbol.endswith(".KS"):
            alt_symbol = symbol.replace(".KS", ".KQ")
            alt_ticker = yf.Ticker(alt_symbol)
            alt_hist = alt_ticker.history(period=period, interval="1d")
            if not alt_hist.empty:
                symbol = alt_symbol
                ticker = alt_ticker
                hist = alt_hist

        if hist.empty and symbol.endswith(".KQ"):
            alt_symbol = symbol.replace(".KQ", ".KS")
            alt_ticker = yf.Ticker(alt_symbol)
            alt_hist = alt_ticker.history(period=period, interval="1d")
            if not alt_hist.empty:
                symbol = alt_symbol
                ticker = alt_ticker
                hist = alt_hist

        if hist.empty:
            return {"error": f"No data found for {symbol}"}

        hist = hist.dropna()
        dates = [idx.strftime("%Y-%m-%d") for idx in hist.index]
        open_prices = hist["Open"].tolist()
        high_prices = hist["High"].tolist()
        low_prices = hist["Low"].tolist()
        close_prices = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()

        display_name = get_display_name(symbol, original_input)
        currency = "KRW" if symbol.endswith(".KS") or symbol.endswith(".KQ") else "USD"

        rsi = calculate_rsi(close_prices)
        ma5 = moving_average(close_prices, 5)
        ma20 = moving_average(close_prices, 20)
        ma60 = moving_average(close_prices, 60)
        ma120 = moving_average(close_prices, 120)

        first = close_prices[0]
        last = close_prices[-1]
        prev = close_prices[-2] if len(close_prices) >= 2 else last

        period_change = ((last - first) / first) * 100
        daily_change = ((last - prev) / prev) * 100

        forecast = ai_momentum_forecast(close_prices)
        news = safe_news_sentiment(ticker, symbol=symbol, display_name=display_name)
        auto_signal = automatic_buy_signal(
            rsi=rsi,
            period_change=period_change,
            daily_change=daily_change,
            forecast_change=forecast["forecast_change_pct"],
            news_score=news["score"],
        )

        return {
            "symbol": symbol.upper(),
            "name": display_name,
            "currency": currency,
            "dates": dates,
            "open": clean_list(open_prices),
            "high": clean_list(high_prices),
            "low": clean_list(low_prices),
            "close": clean_list(close_prices),
            "volume": [int(x) if not is_bad_number(x) else 0 for x in volumes],
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "ma120": ma120,
            "summary": {
                "last_price": round(float(last), 2),
                "daily_change": round(float(daily_change), 2),
                "period_change": round(float(period_change), 2),
                "high_price": round(float(max(high_prices)), 2),
                "low_price": round(float(min(low_prices)), 2),
                "rsi": round(float(rsi), 1),
                "signal": auto_signal["label"],
                "score": auto_signal["score"],
                "score_max": auto_signal.get("score_max", 10),
                "forecast_30d": round(float(forecast["forecast_change_pct"]), 2),
                "forecast_price": round(float(forecast["forecast_price"]), 2),
                "news_sentiment": news["label"],
                "news_score": news["score"],
                "news_score_max": news.get("score_max", 10),
                "news_count": len(news.get("items", [])),
            },
            "analysis": {
                "technical": make_technical_text(display_name, rsi, period_change, daily_change),
                "pattern": make_pattern_text(period_change, rsi),
                "lstm": forecast["text"],
                "news": news["text"],
                "auto_signal": auto_signal["text"],
                "market_summary": f"{display_name}의 현재가는 {round(float(last), 2)}이며, 선택 기간 수익률은 {round(float(period_change), 2)}%입니다. 현재 신호는 '{auto_signal['label']}'입니다.",
            },
            "news": news["items"],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/module/{module_id}")
def get_module(module_id: str, period: str = "6mo"):
    period = validate_period(period)
    if module_id == "market":
        return market_overview(period)
    if module_id == "fundamental":
        return market_fundamental(period)
    if module_id == "signal":
        return market_signal(period)
    if module_id == "macro":
        return macro_monitoring(period)
    if module_id == "sector_valuation":
        return sector_valuation(period)
    if module_id == "sector_momentum":
        return sector_momentum(period)
    if module_id == "market_value":
        return market_value(period)
    return {"error": "Unknown module"}


# =========================================================
# Market / Data Helpers
# =========================================================
def yahoo_market_news_search(query: str):
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": 0, "newsCount": 8, "enableFuzzyQuery": "true"}
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, params=params, headers=headers, timeout=8)
        data = res.json()

        results = []
        for item in data.get("news", []):
            title = item.get("title", "") or ""
            publisher = item.get("publisher", "") or ""
            link = item.get("link", "") or ""
            published = item.get("providerPublishTime", None)
            related = item.get("relatedTickers", [])

            date_text = ""
            if published:
                try:
                    date_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
                except Exception:
                    date_text = ""

            results.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "date": date_text,
                    "related": ", ".join(related[:3]) if isinstance(related, list) else "Market",
                }
            )
        return results
    except Exception:
        return []


def classify_market_news(title: str):
    t = title.lower()
    if any(w in t for w in ["fed", "rate", "inflation", "yield", "treasury"]):
        return "금리·인플레이션"
    if any(w in t for w in ["ai", "nvidia", "semiconductor", "chip", "tech"]):
        return "AI·반도체"
    if any(w in t for w in ["oil", "energy", "crude", "wti", "gas"]):
        return "에너지·원자재"
    if any(w in t for w in ["earnings", "profit", "revenue", "guidance"]):
        return "실적"
    if any(w in t for w in ["nasdaq", "s&p", "dow", "stock market", "market"]):
        return "시장 전체"
    if any(w in t for w in ["korea", "kospi", "won", "samsung"]):
        return "한국시장"
    return "시장 이슈"


def score_market_news(title: str):
    t = title.lower()
    score = 0
    high_keywords = [
        "fed", "inflation", "rate", "nasdaq", "s&p", "ai", "nvidia", "semiconductor", "earnings", "market", "oil", "bond", "yield", "tariff", "china", "recession", "rally", "selloff",
    ]
    medium_keywords = ["stocks", "etf", "dollar", "gold", "korea", "kospi", "growth", "profit", "revenue", "forecast"]
    for word in high_keywords:
        if word in t:
            score += 3
    for word in medium_keywords:
        if word in t:
            score += 1
    return score


def make_news_summary(title: str):
    category = classify_market_news(title)
    if category == "금리·인플레이션":
        return "금리, 물가, 채권금리 변화는 성장주와 위험자산 선호도에 직접적인 영향을 줄 수 있습니다."
    if category == "AI·반도체":
        return "AI와 반도체 관련 이슈는 기술주, 성장주, 관련 공급망 종목의 투자심리에 영향을 줄 수 있습니다."
    if category == "에너지·원자재":
        return "유가와 원자재 가격 변화는 인플레이션, 운송비, 산업재 수익성에 영향을 줄 수 있습니다."
    if category == "실적":
        return "기업 실적과 가이던스는 개별 종목뿐 아니라 해당 섹터의 밸류에이션에도 영향을 줄 수 있습니다."
    if category == "한국시장":
        return "한국시장 관련 이슈는 환율, 외국인 수급, 반도체·자동차·2차전지 섹터와 함께 확인할 필요가 있습니다."
    return "시장 전반의 투자심리와 자금 흐름에 영향을 줄 수 있는 이슈입니다."


def fallback_market_news():
    return [
        {
            "category": "시장 이슈",
            "symbol": "Market",
            "title": "현재 시장 이슈 데이터를 일시적으로 불러오지 못했습니다.",
            "publisher": "Alpharion AI",
            "link": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "잠시 후 다시 시도하면 시장 이슈 뉴스가 표시됩니다.",
            "importance": 0,
        }
    ]


def market_overview(period):
    items = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "KOSPI": "^KS11", "KOSDAQ": "^KQ11", "USD/KRW": "KRW=X", "WTI": "CL=F"}
    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    sentiment = "긍정" if avg > 0.8 else "부정" if avg < -0.8 else "중립"
    return {
        "title": "시황",
        "subtitle": f"{period_label(period)} 기준 주요 지수·환율·원자재 시장 심리 요약",
        "cards": [
            {"label": "시장 심리", "value": sentiment},
            {"label": "평균 변동률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 지표", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시황 변동률 - {period_label(period)}",
        "insight": f"{period_label(period)} 동안 주요 글로벌 지수, 한국 지수, 환율, 원자재 흐름을 기준으로 시장 분위기를 요약했습니다.",
        "analysis_cards": [
            {"title": "시장 심리", "text": f"평균 변동률은 {round(avg, 2)}%이며, 종합 시장 심리는 '{sentiment}'입니다."},
            {"title": "위험 요인", "text": "환율, 유가, 금리성 지표가 동시에 상승하면 위험자산 부담이 커질 수 있습니다."},
            {"title": "확인 포인트", "text": "상승 지표와 하락 지표의 비율을 확인해 단기 시장 방향성을 점검해야 합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "변동률 (%)"},
    }


def market_fundamental(period):
    items = {"S&P 500 ETF": "SPY", "NASDAQ ETF": "QQQ", "Korea ETF": "EWY", "US Value ETF": "VTV", "US Growth ETF": "VUG"}
    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    status = "양호" if avg > 5 else "보통" if avg > -5 else "약화"
    return {
        "title": "펀더멘털",
        "subtitle": f"{period_label(period)} 기준 시장 ETF 기반 체력 진단",
        "cards": [
            {"label": "시장 체력", "value": status},
            {"label": "평균 수익률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 ETF", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 ETF 수익률 - {period_label(period)}",
        "insight": "개별 종목이 아니라 주요 시장 ETF의 기간별 성과를 기준으로 시장의 기본 체력을 진단합니다.",
        "analysis_cards": [
            {"title": "시장 체력", "text": f"{period_label(period)} 기준 평균 수익률은 {round(avg, 2)}%이며, 시장 체력은 '{status}'로 판단됩니다."},
            {"title": "성장/가치 비교", "text": "Growth ETF와 Value ETF의 상대 흐름을 보면 시장 선호 스타일을 확인할 수 있습니다."},
            {"title": "한국시장 위치", "text": "EWY 흐름을 미국 주요 ETF와 비교하면 한국 시장의 상대 강도를 볼 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"},
    }


def market_signal(period):
    items = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "KOSPI": "^KS11", "KOSDAQ": "^KQ11", "Russell 2000": "^RUT"}
    rows, labels, values = build_metric_rows(items, period)
    positive_count = len([v for v in values if v > 0])
    negative_count = len([v for v in values if v < 0])
    signal = "상승 우위" if positive_count > negative_count else "하락 경계" if negative_count > positive_count else "중립"
    return {
        "title": "신호",
        "subtitle": f"{period_label(period)} 기준 시장 지수 상승·하락 신호",
        "cards": [
            {"label": "시장 신호", "value": signal},
            {"label": "상승 지표", "value": positive_count},
            {"label": "하락 지표", "value": negative_count},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 방향성 신호 - {period_label(period)}",
        "insight": "개별 종목 신호가 아니라 주요 시장 지수의 최근 흐름을 기준으로 시장 방향성을 판단합니다.",
        "analysis_cards": [
            {"title": "상승/하락 비율", "text": f"상승 지표는 {positive_count}개, 하락 지표는 {negative_count}개이며 시장 신호는 '{signal}'입니다."},
            {"title": "시장 폭", "text": "여러 지수가 동시에 상승하면 시장 폭이 넓은 상승으로 해석할 수 있습니다."},
            {"title": "주의 구간", "text": "일부 대형 지수만 상승하고 중소형 지수가 약하면 상승 지속성을 확인해야 합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"},
    }


def macro_monitoring(period):
    items = {"US 10Y Yield": "^TNX", "Dollar Index": "DX-Y.NYB", "WTI Oil": "CL=F", "Gold": "GC=F", "USD/KRW": "KRW=X"}
    rows, labels, values = build_metric_rows(items, period)
    risk_score = len([v for v in values if v > 1])
    return {
        "title": "거시경제",
        "subtitle": f"{period_label(period)} 기준 금리·환율·원자재 모니터링",
        "cards": [
            {"label": "Macro Risk", "value": "높음" if risk_score >= 3 else "보통"},
            {"label": "관찰 지표", "value": len(rows)},
            {"label": "상승 지표", "value": risk_score},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"거시경제 지표 변동률 - {period_label(period)}",
        "insight": "금리, 달러, 유가, 금, 환율을 통해 시장의 거시 위험을 점검합니다.",
        "analysis_cards": [
            {"title": "금리 분석", "text": "미국 10년물 금리 상승은 성장주 밸류에이션에 부담을 줄 수 있습니다."},
            {"title": "환율 분석", "text": "USD/KRW 상승은 외국인 수급과 수입물가 부담을 함께 확인해야 합니다."},
            {"title": "원자재 분석", "text": "유가와 금 가격은 인플레이션 및 위험회피 심리 판단에 활용됩니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "변동률 (%)"},
    }


def sector_valuation(period):
    items = sector_items()
    rows, labels, values = build_metric_rows(items, period)
    best = labels[int(np.argmax(values))] if values else "-"
    return {
        "title": "섹터 밸류에이션",
        "subtitle": f"{period_label(period)} 기준 섹터 ETF 상대 성과",
        "cards": [
            {"label": "강세 섹터", "value": best},
            {"label": "관찰 섹터", "value": len(rows)},
            {"label": "평균 수익률", "value": f"{round(safe_mean(values), 2)}%"},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"섹터 상대 성과 - {period_label(period)}",
        "insight": "섹터 ETF의 기간별 성과를 비교해 상대적으로 강한 섹터를 확인합니다.",
        "analysis_cards": [
            {"title": "강세 섹터", "text": f"{period_label(period)} 기준 가장 강한 섹터는 {best}입니다."},
            {"title": "상대 밸류", "text": "가격 성과가 강한 섹터는 이익 기대 또는 자금 유입 가능성을 함께 확인해야 합니다."},
            {"title": "분산 확인", "text": "특정 섹터만 강하면 순환매인지, 구조적 강세인지 추가 확인이 필요합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"},
    }


def sector_momentum(period):
    items = sector_items()
    rows, labels, values = build_metric_rows(items, period)
    ranked = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    leader = ranked[0][0] if ranked else "-"
    return {
        "title": "섹터 모멘텀",
        "subtitle": f"{period_label(period)} 기준 섹터 수익률 랭킹",
        "cards": [
            {"label": "1위 섹터", "value": leader},
            {"label": "관찰 섹터", "value": len(rows)},
            {"label": "평균 모멘텀", "value": f"{round(safe_mean(values), 2)}%"},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"섹터 모멘텀 - {period_label(period)}",
        "insight": "기간별 섹터 ETF 흐름을 기준으로 단기·중기 모멘텀을 측정합니다.",
        "analysis_cards": [
            {"title": "모멘텀 리더", "text": f"{period_label(period)} 기준 모멘텀 1위 섹터는 {leader}입니다."},
            {"title": "순환매 가능성", "text": "기간을 바꾸며 리더 섹터가 바뀌는지 확인하면 순환매 흐름을 볼 수 있습니다."},
            {"title": "추세 지속성", "text": "1개월과 6개월 모두 강한 섹터는 추세 지속 가능성을 더 높게 볼 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"},
    }


def market_value(period):
    items = {"SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM", "EWY": "EWY"}
    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    valuation = "고평가 경계" if avg > 15 else "중립" if avg > -5 else "저평가 가능성"
    return {
        "title": "시장 밸류",
        "subtitle": f"{period_label(period)} 기준 주요 ETF 고·저평가 점검",
        "cards": [
            {"label": "시장 판단", "value": valuation},
            {"label": "평균 수익률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 ETF", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 밸류 점검 - {period_label(period)}",
        "insight": "주요 시장 ETF의 기간별 성과를 기준으로 시장의 고평가·저평가 가능성을 점검합니다.",
        "analysis_cards": [
            {"title": "시장 판단", "text": f"{period_label(period)} 기준 시장 판단은 '{valuation}'입니다."},
            {"title": "과열 확인", "text": "주요 ETF가 장기간 급등한 경우 단기 조정 가능성을 함께 확인해야 합니다."},
            {"title": "저평가 가능성", "text": "장기 하락 이후 회복 신호가 나타나면 저평가 반등 가능성을 점검할 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"},
    }


def sector_items():
    return {
        "Technology": "XLK",
        "Financial": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrial": "XLI",
        "Utilities": "XLU",
    }


def build_metric_rows(items, period):
    rows = []
    labels = []
    values = []
    for name, symbol in items.items():
        metric = quick_metric(symbol, period)
        rows.append({"name": name, "value": metric["text"]})
        labels.append(name)
        values.append(metric["change"])
    return rows, labels, values


def quick_metric(symbol, period="1mo"):
    try:
        hist = yf.Ticker(symbol).history(period=period, interval="1d").dropna()
        if hist.empty:
            return {"change": 0, "text": "데이터 없음"}
        first = float(hist["Close"].iloc[0])
        last = float(hist["Close"].iloc[-1])
        change = ((last - first) / first) * 100
        return {"change": round(change, 2), "text": f"{round(last, 2)} / {round(change, 2)}%"}
    except Exception:
        return {"change": 0, "text": "데이터 없음"}


def get_krx_stocks():
    global KRX_CACHE
    if KRX_CACHE is not None:
        return KRX_CACHE

    stocks = []
    seen = set()
    if not FDR_AVAILABLE:
        KRX_CACHE = stocks
        return stocks

    for target in ["KRX", "ETF/KR"]:
        try:
            df = fdr.StockListing(target)
            for _, row in df.iterrows():
                name = str(row.get("Name", "") or row.get("NameEng", "") or row.get("Symbol", "")).strip()
                code = str(row.get("Code", "") or row.get("Symbol", "")).strip()
                market = str(row.get("Market", "") or target).strip()
                if not name or not code:
                    continue

                code = code.zfill(6) if code.isdigit() and len(code) < 6 else code
                symbol = code + ".KQ" if market == "KOSDAQ" else code + ".KS"
                key = f"{name}-{symbol}"
                if key in seen:
                    continue
                seen.add(key)
                stocks.append({"name": name, "symbol": symbol, "market": market or "Korea", "type": "ETF" if target == "ETF/KR" else "EQUITY"})
        except Exception:
            continue

    KRX_CACHE = stocks
    return stocks


def search_krx_by_name(q: str):
    q_norm = normalize_text(q)
    results = []
    if not q_norm:
        return results

    for item in get_krx_stocks():
        name = item["name"]
        symbol = item["symbol"]
        pure_code = symbol.replace(".KS", "").replace(".KQ", "")
        name_norm = normalize_text(name)
        symbol_norm = normalize_text(symbol)
        code_norm = normalize_text(pure_code)

        if q_norm in name_norm or name_norm in q_norm or q_norm in symbol_norm or q_norm in code_norm or code_norm in q_norm:
            results.append(item)
        if len(results) >= 30:
            break
    return results


def yahoo_search(q: str):
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {"q": q, "quotesCount": 20, "newsCount": 0, "enableFuzzyQuery": "true"}
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, params=params, headers=headers, timeout=8)
        data = res.json()
        results = []
        for item in data.get("quotes", []):
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or item.get("name")
            exchange = item.get("exchange") or item.get("exchDisp") or "Unknown"
            quote_type = item.get("quoteType", "")
            if symbol:
                results.append({"name": name or symbol, "symbol": symbol, "market": exchange, "type": quote_type})
        return results
    except Exception:
        return []


def validate_period(period: str):
    allowed = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
    return period if period in allowed else "1y"


def period_label(period: str):
    labels = {"1mo": "1개월", "3mo": "3개월", "6mo": "6개월", "1y": "1년", "2y": "2년", "5y": "5년", "10y": "10년", "max": "전체 기간"}
    return labels.get(period, "1년")


def normalize_symbol(value: str):
    value = value.strip()
    if value in KOREAN_NAME_MAP:
        return KOREAN_NAME_MAP[value]

    value_norm = normalize_text(value)
    for name, code in KOREAN_NAME_MAP.items():
        if value_norm == normalize_text(name):
            return code

    if "(" in value and ")" in value:
        start = value.find("(") + 1
        end = value.find(")")
        inside = value[start:end].strip()
        krx_results = search_krx_by_name(inside)
        if krx_results:
            return krx_results[0]["symbol"]
        if "." in inside:
            return inside.upper()
        value_without_paren = value.split("(")[0].strip()
        krx_results = search_krx_by_name(value_without_paren)
        if krx_results:
            return krx_results[0]["symbol"]
        return inside.upper()

    krx_results = search_krx_by_name(value)
    if krx_results:
        return krx_results[0]["symbol"]

    if value.isdigit() and len(value) == 6:
        return value + ".KS"
    if contains_korean(value):
        searched = yahoo_search(value)
        if searched:
            return searched[0]["symbol"]
    return value.upper()


def get_display_name(symbol: str, original_input: str = ""):
    original_input = original_input.strip()
    if "(" in original_input and ")" in original_input:
        name_part = original_input.split("(")[0].strip()
        if name_part:
            return name_part
    if original_input in KOREAN_NAME_MAP:
        return original_input
    for name, code in KOREAN_NAME_MAP.items():
        if code == symbol:
            return name
    for item in get_krx_stocks():
        if item["symbol"] == symbol:
            return item["name"]
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("shortName") or info.get("longName") or symbol
    except Exception:
        return symbol


def normalize_text(text: str):
    return str(text or "").replace(" ", "").replace("-", "").replace("_", "").replace("/", "").replace(".", "").upper().strip()


def contains_korean(text: str):
    return any("가" <= ch <= "힣" for ch in text)


def is_bad_number(x):
    try:
        return x is None or math.isnan(float(x)) or math.isinf(float(x))
    except Exception:
        return True


def clean_list(values):
    cleaned = []
    for x in values:
        if is_bad_number(x):
            cleaned.append(None)
        else:
            cleaned.append(round(float(x), 2))
    return cleaned


def moving_average(values, window):
    result = []
    for i in range(len(values)):
        if i + 1 < window:
            result.append(None)
        else:
            avg = np.mean(values[i + 1 - window : i + 1])
            result.append(round(float(avg), 2))
    return result


def calculate_rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ai_momentum_forecast(values):
    last_price = float(values[-1])
    recent = values[-30:] if len(values) >= 30 else values
    if len(recent) < 2:
        return {"forecast_price": last_price, "forecast_change_pct": 0.0, "text": "데이터가 부족하여 예측을 보류합니다."}

    momentum = ((recent[-1] - recent[0]) / recent[0]) * 100
    volatility = np.std(np.diff(recent)) / np.mean(recent) * 100
    forecast_change = momentum * 0.55 - volatility * 0.2
    forecast_price = last_price * (1 + forecast_change / 100)
    return {
        "forecast_price": forecast_price,
        "forecast_change_pct": forecast_change,
        "text": f"AI 기반 30일 예측은 최근 가격 모멘텀과 변동성을 반영했습니다. 예상 가격은 약 {forecast_price:.2f}, 현재가 대비 예상 변화율은 {forecast_change:.2f}%입니다.",
    }


def safe_news_sentiment(ticker, symbol: str = "", display_name: str = ""):
    try:
        return news_sentiment(ticker, symbol=symbol, display_name=display_name)
    except Exception as e:
        print("NEWS SENTIMENT ERROR:", repr(e))
        return {
            "score": 5,
            "score_max": 10,
            "raw_score": 0,
            "label": "중립",
            "items": [],
            "text": "뉴스 데이터를 충분히 가져오지 못해 감성 점수는 중립 5점/10점으로 처리했습니다.",
        }


def _extract_yfinance_news_items(ticker):
    """
    yfinance 버전별 뉴스 구조 차이를 모두 처리합니다.
    일부 환경에서는 ticker.news의 title이 최상위가 아니라 content 안에 들어옵니다.
    """
    items = []
    try:
        raw_news = ticker.news or []
    except Exception:
        raw_news = []

    for n in raw_news[:12]:
        title = ""
        publisher = ""
        link = ""
        published = None

        if isinstance(n, dict):
            title = n.get("title") or ""
            publisher = n.get("publisher") or n.get("provider") or ""
            link = n.get("link") or n.get("url") or ""
            published = n.get("providerPublishTime") or n.get("pubDate") or n.get("displayTime")

            content = n.get("content")
            if isinstance(content, dict):
                title = title or content.get("title") or content.get("headline") or ""
                provider = content.get("provider")
                if isinstance(provider, dict):
                    publisher = publisher or provider.get("displayName") or provider.get("name") or ""
                elif isinstance(provider, str):
                    publisher = publisher or provider

                canonical_url = content.get("canonicalUrl")
                if isinstance(canonical_url, dict):
                    link = link or canonical_url.get("url") or ""
                elif isinstance(canonical_url, str):
                    link = link or canonical_url

                click_through_url = content.get("clickThroughUrl")
                if not link and isinstance(click_through_url, dict):
                    link = click_through_url.get("url") or ""

                published = published or content.get("pubDate") or content.get("displayTime")

        if title:
            date_text = ""
            try:
                if isinstance(published, (int, float)):
                    date_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
                elif isinstance(published, str) and published:
                    date_text = published[:10]
            except Exception:
                date_text = ""

            items.append({
                "title": title,
                "publisher": publisher or "Market News",
                "link": link,
                "date": date_text,
            })

    return items


def _extract_search_news_items(symbol: str, display_name: str):
    """
    ticker.news가 비어 있거나 0점만 나오는 경우를 막기 위해
    Yahoo Finance Search 뉴스 결과를 보조 뉴스 소스로 사용합니다.
    """
    items = []
    queries = []

    clean_symbol = str(symbol or "").strip()
    clean_name = str(display_name or "").strip()

    if clean_symbol:
        queries.append(f"{clean_symbol} stock news earnings outlook")
    if clean_name and normalize_text(clean_name) != normalize_text(clean_symbol):
        queries.append(f"{clean_name} stock news earnings outlook")
    if clean_symbol or clean_name:
        queries.append(f"{clean_symbol or clean_name} market news")

    seen = set()
    for q in queries:
        try:
            for item in yahoo_market_news_search(q):
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                key = normalize_text(title)
                if key in seen:
                    continue
                seen.add(key)
                items.append({
                    "title": title,
                    "publisher": item.get("publisher") or "Yahoo Finance",
                    "link": item.get("link") or "",
                    "date": item.get("date") or "",
                })
                if len(items) >= 12:
                    return items
        except Exception as e:
            print("SEARCH NEWS ERROR:", repr(e))
            continue

    return items


def news_sentiment(ticker, symbol: str = "", display_name: str = ""):
    """
    뉴스 감성 점수:
    - 0점 ~ 10점 만점
    - 5점 = 중립
    - 7점 이상 = 긍정
    - 3점 이하 = 부정
    """
    positive_words = [
        "beat", "beats", "growth", "strong", "surge", "record", "upgrade", "upgraded",
        "profit", "profits", "bullish", "gain", "gains", "rally", "outperform", "buy",
        "raised", "raises", "demand", "ai", "launch", "partnership", "contract",
        "approval", "expands", "expansion", "tops", "higher", "optimistic", "positive",
        "accelerate", "breakthrough", "rebound", "jump", "jumps", "soar", "soars"
    ]

    negative_words = [
        "miss", "misses", "fall", "falls", "drop", "drops", "weak", "downgrade",
        "downgraded", "loss", "losses", "bearish", "risk", "lawsuit", "cut", "cuts",
        "slowdown", "concern", "concerns", "lower", "probe", "investigation",
        "warning", "slump", "plunge", "decline", "negative", "delay", "recall",
        "selloff", "sell-off", "down", "fraud", "ban", "tariff"
    ]

    items = _extract_yfinance_news_items(ticker)

    if len(items) < 3:
        items = items + _extract_search_news_items(symbol, display_name)

    deduped = []
    seen = set()
    for item in items:
        title = item.get("title", "")
        key = normalize_text(title)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    items = deduped[:12]

    raw_score = 0.0
    scored_items = []

    for item in items:
        title = item.get("title", "") or ""
        title_lower = title.lower()

        item_score = 0.0

        for w in positive_words:
            if w in title_lower:
                item_score += 1.0
        for w in negative_words:
            if w in title_lower:
                item_score -= 1.0

        # 제목에 직접적인 긍정/부정 단어가 없어도 주요 성장/리스크 키워드를 약하게 반영
        if item_score == 0:
            if any(x in title_lower for x in ["nvidia", "semiconductor", "cloud", "earnings", "revenue", "data center", "gpu"]):
                item_score += 0.5
            if any(x in title_lower for x in ["fed", "rates", "inflation", "tariff", "china", "yield"]):
                item_score -= 0.5

        raw_score += item_score
        scored_items.append({**item, "sentiment_score": round(float(item_score), 2)})

    if not items:
        return {
            "score": 5,
            "score_max": 10,
            "raw_score": 0,
            "label": "중립",
            "items": [],
            "text": "최근 뉴스 데이터가 부족하여 감성 점수는 중립 5점/10점으로 처리했습니다.",
        }

    normalized = 5 + raw_score
    normalized = max(0, min(10, round(float(normalized), 1)))

    if normalized >= 7:
        label = "긍정"
    elif normalized <= 3:
        label = "부정"
    else:
        label = "중립"

    return {
        "score": normalized,
        "score_max": 10,
        "raw_score": round(float(raw_score), 2),
        "label": label,
        "items": scored_items,
        "text": f"최근 뉴스 헤드라인 {len(items)}건 기준 감성 점수는 {normalized}점/10점 만점이며, 종합 판단은 '{label}'입니다.",
    }


def automatic_buy_signal(rsi, period_change, daily_change, forecast_change, news_score):
    """
    자동 매수 신호:
    - 최종 점수는 0점 ~ 10점 만점
    - 5점 부근 = 관망
    - 6점 이상 = 매수 관심
    - 8점 이상 = 강한 매수 관심
    """
    raw = 0

    # RSI: 최대 +2 / 최소 -2
    if rsi < 30:
        raw += 2
    elif 30 <= rsi < 45:
        raw += 1
    elif 45 <= rsi <= 65:
        raw += 1
    elif rsi > 75:
        raw -= 2
    elif rsi > 70:
        raw -= 1

    # 기간 수익률: 최대 +2 / 최소 -2
    if period_change > 20:
        raw += 2
    elif period_change > 5:
        raw += 1
    elif period_change < -20:
        raw -= 2
    elif period_change < -8:
        raw -= 1

    # 단기 변동률: 최대 +1 / 최소 -1
    if daily_change > 0:
        raw += 1
    elif daily_change < -3:
        raw -= 1

    # AI 30일 예측: 최대 +2 / 최소 -2
    if forecast_change > 10:
        raw += 2
    elif forecast_change > 0:
        raw += 1
    elif forecast_change < -10:
        raw -= 2
    elif forecast_change < 0:
        raw -= 1

    # 뉴스 감성: 최대 +2 / 최소 -2
    ns = float(news_score if news_score is not None else 5)
    if ns >= 8:
        raw += 2
    elif ns >= 6:
        raw += 1
    elif ns <= 2:
        raw -= 2
    elif ns <= 4:
        raw -= 1

    final_score = round(max(0, min(10, 5 + raw * 0.75)), 1)

    if final_score >= 8:
        label = "강한 매수 관심"
    elif final_score >= 6:
        label = "매수 관심"
    elif final_score <= 3:
        label = "매수 보류"
    else:
        label = "관망"

    return {
        "score": final_score,
        "score_max": 10,
        "raw_score": raw,
        "label": label,
        "text": f"자동 매수 신호 점수는 {final_score}점/10점 만점입니다. RSI, 기간 수익률, 단기 변동률, AI 30일 예측, 뉴스 감성 점수를 종합하여 '{label}'로 판단했습니다.",
    }



def make_technical_text(name, rsi, period_change, daily_change):
    return f"{name}는 선택 기간 기준 {period_change:.2f}% 변동했습니다. RSI는 {rsi:.1f}이며, 직전 거래일 대비 변동률은 {daily_change:.2f}%입니다."


def make_pattern_text(period_change, rsi):
    if period_change > 10:
        return "선택 기간 동안 우상향 흐름이 나타납니다. 추세 지속형 패턴 또는 신고가 돌파 가능성을 확인해야 합니다."
    if period_change < -10:
        return "약세 흐름이 나타납니다. 지지선 이탈 여부와 거래량 증가 여부를 확인해야 합니다."
    if 45 <= rsi <= 60:
        return "강한 방향성보다는 박스권 또는 횡보 패턴 가능성이 있습니다."
    return "현재 구간은 뚜렷한 패턴보다 변동성 확인이 우선입니다."


def safe_mean(values):
    try:
        return float(np.mean(values)) if values else 0
    except Exception:
        return 0
