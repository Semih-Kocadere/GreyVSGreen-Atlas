# backend/create_database.py
from getpass import getpass
from typing import Optional
from passlib.context import CryptContext
from sqlmodel import SQLModel, Field, create_engine, Session, select
from pydantic import EmailStr
import argparse
import os

DB_URL = os.getenv("DB_URL", "sqlite:///./db.sqlite")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: EmailStr
    password_hash: str
    full_name: Optional[str] = None
    is_active: bool = True

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def user_exists(email: str) -> bool:
    with Session(engine) as s:
        return s.exec(select(User).where(User.email == email)).first() is not None

def add_user(email: str, password: str, full_name: Optional[str] = None):
    if user_exists(email):
        print(f"[!] Zaten kayıtlı: {email}")
        return
    with Session(engine) as s:
        u = User(email=email, password_hash=pwd_context.hash(password), full_name=full_name)
        s.add(u); s.commit()
        print(f"[✓] Eklendi: {email}")

def main():
    parser = argparse.ArgumentParser(description="Create DB and add users")
    parser.add_argument("--email", help="Kullanıcı email")
    parser.add_argument("--name", help="Ad Soyad", default=None)
    parser.add_argument("--password", help="Şifre (opsiyonel, boşsa sorulur)", default=None)
    parser.add_argument("--demo", action="store_true", help="demo@example.com / demo123 ekle")
    args = parser.parse_args()

    create_db_and_tables()
    print("[✓] DB ve tablolar hazır")

    if args.demo:
        add_user("demo@example.com", "demo123", "Demo User")
        return

    if not args.email:
        print("Örnek kullanım:")
        print("  python -m backend.create_database --email user@example.com --name 'Ad Soyad'")
        return

    pwd = args.password or getpass("Şifre: ")
    add_user(args.email, pwd, args.name)

if __name__ == "__main__":
    main()
