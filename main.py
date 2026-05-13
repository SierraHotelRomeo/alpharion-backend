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
from typing import Optional

import numpy as np
import requests
import yfinance as yf
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

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

# =========================================================
# Plan / NICEPAY Payment Settings
# =========================================================
FREE_AI_ANALYSIS_LIMIT = int(os.getenv("FREE_AI_ANALYSIS_LIMIT", "5"))
STANDARD_PLAN_DAYS = int(os.getenv("STANDARD_PLAN_DAYS", "365"))
STANDARD_PLAN_AMOUNT = int(os.getenv("STANDARD_PLAN_AMOUNT", "99000"))
STANDARD_PLAN_NAME = os.getenv("STANDARD_PLAN_NAME", "Alpharion Standard 1년 이용권")

# NICEPAY 개발자센터에서 발급받은 값만 사용합니다.
# NICEPAY_CLIENT_ID: 결제창 호출용 클라이언트키
# NICEPAY_SECRET_KEY: 서버 승인용 시크릿키
# 기존에 NICEPAY_MERCHANT_KEY 이름으로 넣어둔 경우도 자동 호환됩니다.
NICEPAY_CLIENT_ID = os.getenv("NICEPAY_CLIENT_ID", "")
NICEPAY_SECRET_KEY = os.getenv("NICEPAY_SECRET_KEY", "") or os.getenv("NICEPAY_MERCHANT_KEY", "")
NICEPAY_API_BASE = os.getenv("NICEPAY_API_BASE", "https://api.nicepay.co.kr")
NICEPAY_SUCCESS_CODES = {"0000"}

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


class AdminPlanUpdateRequest(BaseModel):
    plan_type: str
    plan_days: Optional[int] = STANDARD_PLAN_DAYS
    reset_usage: bool = False


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

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payment_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            plan_type TEXT NOT NULL DEFAULT 'STANDARD',
            status TEXT NOT NULL DEFAULT 'READY',
            nicepay_tid TEXT,
            pay_method TEXT,
            result_code TEXT,
            result_msg TEXT,
            raw_response TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # 기존 서버 DB를 그대로 쓰는 경우를 위한 안전 마이그레이션
    ensure_column(conn, "users", "plan_type", "plan_type TEXT DEFAULT 'FREE'")
    ensure_column(conn, "users", "plan_expire_at", "plan_expire_at TEXT")
    ensure_column(conn, "users", "ai_analysis_count", "ai_analysis_count INTEGER DEFAULT 0")
    ensure_column(conn, "users", "last_ai_analysis_at", "last_ai_analysis_at TEXT")
    ensure_column(conn, "users", "recovery_code_hash", "recovery_code_hash TEXT")
    ensure_column(conn, "users", "terms_accepted", "terms_accepted INTEGER DEFAULT 0")
    ensure_column(conn, "users", "password_plain", "password_plain TEXT")
    ensure_column(conn, "users", "find_code_hash", "find_code_hash TEXT")
    ensure_column(conn, "users", "find_code_expires_at", "find_code_expires_at TEXT")
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


def get_effective_plan_type(user):
    if not user:
        return "FREE"
    plan_type = (user["plan_type"] or "FREE").upper() if "plan_type" in user.keys() else "FREE"
    plan_expire_at = user["plan_expire_at"] if "plan_expire_at" in user.keys() else None

    if plan_type == "STANDARD" and plan_expire_at:
        try:
            if utcnow() > datetime.fromisoformat(plan_expire_at):
                downgrade_user_to_free(user["id"])
                return "FREE"
        except Exception:
            downgrade_user_to_free(user["id"])
            return "FREE"
    return plan_type if plan_type in {"FREE", "STANDARD"} else "FREE"


def public_user(user):
    if not user:
        return None
    plan_type = get_effective_plan_type(user)
    count = int(user["ai_analysis_count"] or 0) if "ai_analysis_count" in user.keys() else 0
    remaining = None if plan_type == "STANDARD" else max(FREE_AI_ANALYSIS_LIMIT - count, 0)
    return {
        "id": user["id"],
        "email": user["email"],
        "created_at": user["created_at"],
        "plan_type": plan_type,
        "plan_expire_at": user["plan_expire_at"] if "plan_expire_at" in user.keys() else None,
        "ai_analysis_count": count,
        "free_ai_limit": FREE_AI_ANALYSIS_LIMIT,
        "ai_remaining": remaining,
    }


def downgrade_user_to_free(user_id: int):
    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET plan_type='FREE', plan_expire_at=NULL, updated_at=?
        WHERE id=?
        """,
        (utcnow().isoformat(), user_id),
    )
    conn.commit()
    conn.close()


def activate_standard_plan(user_id: int, days: int = STANDARD_PLAN_DAYS):
    expire_at = utcnow() + timedelta(days=days)
    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET plan_type='STANDARD', plan_expire_at=?, updated_at=?
        WHERE id=?
        """,
        (expire_at.isoformat(), utcnow().isoformat(), user_id),
    )
    conn.commit()
    conn.close()
    return expire_at


def check_and_consume_ai_usage(user):
    plan_type = get_effective_plan_type(user)
    if plan_type == "STANDARD":
        return

    count = int(user["ai_analysis_count"] or 0) if "ai_analysis_count" in user.keys() else 0
    if count >= FREE_AI_ANALYSIS_LIMIT:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FREE_LIMIT_EXCEEDED",
                "message": "Free 요금제의 AI 종목분석 5회 사용을 모두 소진했습니다.",
                "limit": FREE_AI_ANALYSIS_LIMIT,
                "used": count,
                "payment_url": "/payment.html",
            },
        )

    conn = get_auth_db()
    conn.execute(
        """
        UPDATE users
        SET ai_analysis_count=COALESCE(ai_analysis_count, 0)+1, last_ai_analysis_at=?, updated_at=?
        WHERE id=?
        """,
        (utcnow().isoformat(), utcnow().isoformat(), user["id"]),
    )
    conn.commit()
    conn.close()


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


def admin_public_user(row):
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "terms_accepted": bool(row["terms_accepted"]),
        "plan_type": row["plan_type"] if "plan_type" in row.keys() else "FREE",
        "plan_expire_at": row["plan_expire_at"] if "plan_expire_at" in row.keys() else None,
        "ai_analysis_count": int(row["ai_analysis_count"] or 0) if "ai_analysis_count" in row.keys() else 0,
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
            SELECT id, email, terms_accepted, created_at, updated_at, plan_type, plan_expire_at, ai_analysis_count
            FROM users
            WHERE lower(email) LIKE ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (like, limit, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
        rows = conn.execute(
            """
            SELECT id, email, terms_accepted, created_at, updated_at, plan_type, plan_expire_at, ai_analysis_count
            FROM users
            ORDER BY id DESC
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


# =========================================================
# NICEPAY Payment API
# =========================================================
def nicepay_sha256(text: str):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nicepay_basic_auth_header():
    # NICEPAY Basic 인증은 시크릿키 뒤에 콜론(:)을 붙인 값을 base64 인코딩합니다.
    # 예: Authorization: Basic base64(secretKey + ":")
    raw = f"{NICEPAY_SECRET_KEY}:".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("utf-8")


def nicepay_is_configured():
    return bool(NICEPAY_CLIENT_ID and NICEPAY_SECRET_KEY)


def make_order_id(user_id: int):
    return f"AMW{user_id}{utcnow().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(3).upper()}"


def get_payment_order(order_id: str):
    conn = get_auth_db()
    row = conn.execute("SELECT * FROM payment_orders WHERE order_id=?", (order_id,)).fetchone()
    conn.close()
    return row


@app.get("/api/payments/nicepay/config")
def nicepay_config():
    return {
        "ok": True,
        "configured": nicepay_is_configured(),
        "amount": STANDARD_PLAN_AMOUNT,
        "goods_name": STANDARD_PLAN_NAME,
        "plan_days": STANDARD_PLAN_DAYS,
    }


@app.post("/api/payments/nicepay/prepare")
def nicepay_prepare(user=Depends(get_current_user)):
    if not nicepay_is_configured():
        raise HTTPException(status_code=500, detail="NICEPAY 환경변수가 설정되지 않았습니다. NICEPAY_CLIENT_ID, NICEPAY_SECRET_KEY를 확인하세요.")

    order_id = make_order_id(user["id"])
    now = utcnow().isoformat()
    conn = get_auth_db()
    conn.execute(
        """
        INSERT INTO payment_orders (order_id, user_id, amount, plan_type, status, created_at, updated_at)
        VALUES (?, ?, ?, 'STANDARD', 'READY', ?, ?)
        """,
        (order_id, user["id"], STANDARD_PLAN_AMOUNT, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "client_id": NICEPAY_CLIENT_ID,
        "method": "card",
        "order_id": order_id,
        "amount": STANDARD_PLAN_AMOUNT,
        "goods_name": STANDARD_PLAN_NAME,
        "return_url": f"{API_PUBLIC_BASE}/api/payments/nicepay/auth-callback",
        "buyer_name": user["email"].split("@")[0],
        "buyer_email": user["email"],
        "mall_user_id": str(user["id"]),
    }


@app.post("/api/payments/nicepay/auth-callback", response_class=HTMLResponse)
async def nicepay_auth_callback(request: Request):
    form = await request.form()
    data = {k: str(v) for k, v in form.items()}

    # NICEPAY v1 JS SDK는 소문자 camelCase로 응답합니다.
    # 구버전/테스트 응답 호환을 위해 일부 대문자 파라미터도 같이 처리합니다.
    auth_result_code = data.get("authResultCode") or data.get("AuthResultCode") or ""
    auth_result_msg = data.get("authResultMsg") or data.get("AuthResultMsg") or ""
    auth_token = data.get("authToken") or data.get("AuthToken") or ""
    client_id = data.get("clientId") or data.get("MID") or ""
    order_id = data.get("orderId") or data.get("Moid") or ""
    amount = data.get("amount") or data.get("Amt") or "0"
    signature = data.get("signature") or data.get("Signature") or ""
    tid = data.get("tid") or data.get("TxTid") or data.get("TID") or ""
    pay_method = data.get("payMethod") or data.get("PayMethod") or ""

    def redirect_html(status: str, message: str):
        from urllib.parse import urlencode
        qs = urlencode({"status": status, "message": message})
        target = f"{FRONTEND_BASE_URL}/payment-result.html?{qs}"
        return f"""
        <!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">
        <meta http-equiv=\"refresh\" content=\"0; url={target}\">
        <title>Payment Result</title></head>
        <body><script>location.href={json.dumps(target)};</script></body></html>
        """

    order = get_payment_order(order_id)
    if not order:
        return redirect_html("fail", "주문정보를 찾을 수 없습니다.")

    if auth_result_code != "0000":
        conn = get_auth_db()
        conn.execute(
            "UPDATE payment_orders SET status='AUTH_FAILED', result_code=?, result_msg=?, raw_response=?, updated_at=? WHERE order_id=?",
            (auth_result_code, auth_result_msg, json.dumps(data, ensure_ascii=False), utcnow().isoformat(), order_id),
        )
        conn.commit()
        conn.close()
        return redirect_html("fail", auth_result_msg or "결제 인증에 실패했습니다.")

    try:
        amount_int = int(str(amount).replace(",", ""))
    except Exception:
        amount_int = 0

    if client_id != NICEPAY_CLIENT_ID or amount_int != int(order["amount"]):
        return redirect_html("fail", "결제 금액 또는 클라이언트키 정보가 일치하지 않습니다.")

    expected_signature = nicepay_sha256(auth_token + client_id + str(amount_int) + NICEPAY_SECRET_KEY)
    if signature and signature.lower() != expected_signature.lower():
        return redirect_html("fail", "결제 인증 위변조 검증에 실패했습니다.")

    if not tid:
        return redirect_html("fail", "결제 승인키가 없습니다.")

    approve_url = f"{NICEPAY_API_BASE.rstrip('/')}/v1/payments/{tid}"
    approve_payload = {"amount": amount_int}
    headers = {
        "Authorization": nicepay_basic_auth_header(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        res = requests.post(approve_url, headers=headers, json=approve_payload, timeout=25)
        try:
            approve_data = res.json()
        except Exception:
            approve_data = {"raw": res.text, "status_code": res.status_code}
    except Exception as e:
        conn = get_auth_db()
        conn.execute(
            "UPDATE payment_orders SET status='APPROVE_ERROR', result_msg=?, raw_response=?, updated_at=? WHERE order_id=?",
            (str(e), json.dumps(data, ensure_ascii=False), utcnow().isoformat(), order_id),
        )
        conn.commit()
        conn.close()
        return redirect_html("fail", "결제 승인 요청 중 오류가 발생했습니다.")

    result_code = str(approve_data.get("resultCode", approve_data.get("ResultCode", "")))
    result_msg = str(approve_data.get("resultMsg", approve_data.get("ResultMsg", "")))
    approve_amount = int(approve_data.get("amount", approve_data.get("Amt", amount_int)) or 0)
    approve_order_id = str(approve_data.get("orderId", approve_data.get("Moid", order_id)))
    approve_tid = str(approve_data.get("tid", approve_data.get("TID", tid)))
    approve_status = str(approve_data.get("status", ""))
    approve_pay_method = str(approve_data.get("payMethod", pay_method))

    if result_code in NICEPAY_SUCCESS_CODES and approve_status == "paid" and approve_order_id == order_id and approve_amount == int(order["amount"]):
        expire_at = activate_standard_plan(int(order["user_id"]), STANDARD_PLAN_DAYS)
        conn = get_auth_db()
        conn.execute(
            """
            UPDATE payment_orders
            SET status='PAID', nicepay_tid=?, pay_method=?, result_code=?, result_msg=?, raw_response=?, updated_at=?
            WHERE order_id=?
            """,
            (approve_tid, approve_pay_method, result_code, result_msg, json.dumps(approve_data, ensure_ascii=False), utcnow().isoformat(), order_id),
        )
        conn.commit()
        conn.close()
        return redirect_html("success", f"STANDARD 요금제가 활성화되었습니다. 만료일: {expire_at.strftime('%Y-%m-%d')}")

    conn = get_auth_db()
    conn.execute(
        """
        UPDATE payment_orders
        SET status='APPROVE_FAILED', nicepay_tid=?, pay_method=?, result_code=?, result_msg=?, raw_response=?, updated_at=?
        WHERE order_id=?
        """,
        (approve_tid, approve_pay_method, result_code, result_msg, json.dumps(approve_data, ensure_ascii=False), utcnow().isoformat(), order_id),
    )
    conn.commit()
    conn.close()
    return redirect_html("fail", result_msg or "결제 승인에 실패했습니다.")


@app.post("/api/admin/users/{user_id}/plan")
def admin_update_user_plan(user_id: int, req: AdminPlanUpdateRequest, admin_ok=Depends(verify_admin_secret)):
    plan_type = (req.plan_type or "FREE").upper()
    if plan_type not in {"FREE", "STANDARD"}:
        raise HTTPException(status_code=400, detail="plan_type은 FREE 또는 STANDARD만 가능합니다.")

    conn = get_auth_db()
    user = conn.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="해당 회원을 찾을 수 없습니다.")

    if plan_type == "STANDARD":
        days = int(req.plan_days or STANDARD_PLAN_DAYS)
        expire_at = utcnow() + timedelta(days=days)
        conn.execute(
            "UPDATE users SET plan_type='STANDARD', plan_expire_at=?, updated_at=? WHERE id=?",
            (expire_at.isoformat(), utcnow().isoformat(), user_id),
        )
    else:
        conn.execute(
            "UPDATE users SET plan_type='FREE', plan_expire_at=NULL, updated_at=? WHERE id=?",
            (utcnow().isoformat(), user_id),
        )

    if req.reset_usage:
        conn.execute("UPDATE users SET ai_analysis_count=0, last_ai_analysis_at=NULL WHERE id=?", (user_id,))

    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return {"ok": True, "user": public_user(row)}


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
        news = safe_news_sentiment(ticker)
        auto_signal = automatic_buy_signal(
            rsi=rsi,
            period_change=period_change,
            daily_change=daily_change,
            forecast_change=forecast["forecast_change_pct"],
            news_score=news["score"],
        )

        check_and_consume_ai_usage(user)

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
                "forecast_30d": round(float(forecast["forecast_change_pct"]), 2),
                "forecast_price": round(float(forecast["forecast_price"]), 2),
                "news_sentiment": news["label"],
                "news_score": news["score"],
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


def safe_news_sentiment(ticker):
    try:
        return news_sentiment(ticker)
    except Exception:
        return {"score": 0, "label": "중립", "items": [], "text": "뉴스 데이터를 가져오지 못했습니다."}


def news_sentiment(ticker):
    positive_words = ["beat", "growth", "strong", "surge", "record", "upgrade", "profit", "bullish", "gain", "ai", "demand"]
    negative_words = ["miss", "fall", "drop", "weak", "downgrade", "loss", "bearish", "risk", "lawsuit", "cut", "slowdown", "concern"]

    try:
        news_list = ticker.news or []
    except Exception:
        news_list = []

    items = []
    score = 0
    for n in news_list[:8]:
        title = n.get("title", "") or ""
        publisher = n.get("publisher", "") or ""
        link = n.get("link", "") or ""
        published = n.get("providerPublishTime", None)
        title_lower = title.lower()

        for w in positive_words:
            if w in title_lower:
                score += 1
        for w in negative_words:
            if w in title_lower:
                score -= 1

        date_text = ""
        if published:
            try:
                date_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
            except Exception:
                date_text = ""
        if title:
            items.append({"title": title, "publisher": publisher, "link": link, "date": date_text})

    label = "긍정" if score >= 2 else "부정" if score <= -2 else "중립"
    return {"score": score, "label": label, "items": items, "text": f"최근 뉴스 헤드라인 기준 감성 점수는 {score}점이며, 종합 판단은 '{label}'입니다."}


def automatic_buy_signal(rsi, period_change, daily_change, forecast_change, news_score):
    score = 0
    if rsi < 35:
        score += 2
    elif 35 <= rsi <= 60:
        score += 1
    elif rsi > 75:
        score -= 2

    if period_change > 8:
        score += 2
    elif period_change > 3:
        score += 1
    elif period_change < -10:
        score -= 2

    score += 1 if daily_change > 0 else -1

    if forecast_change > 5:
        score += 2
    elif forecast_change > 0:
        score += 1
    elif forecast_change < -5:
        score -= 2

    if news_score >= 2:
        score += 1
    elif news_score <= -2:
        score -= 1

    if score >= 6:
        label = "강한 매수 관심"
    elif score >= 3:
        label = "매수 관심"
    elif score <= -3:
        label = "매수 보류"
    else:
        label = "관망"

    return {"score": score, "label": label, "text": f"자동 매수 신호 점수는 {score}점입니다. RSI, 기간 수익률, 단기 변동률, AI 30일 예측, 뉴스 감성 점수를 종합하여 '{label}'로 판단했습니다."}


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
