# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging

from .api_routes import router

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="🤖 LucasAI Fine-tuning API",
    description="CSV 업로드로 LucasAI 모델을 파인튜닝",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 새로운 workspace 구조 초기화
from .utils import migrate_legacy_structure
from .path_manager import path_manager

# 기존 파일 구조를 새 구조로 마이그레이션
migrate_legacy_structure()

# 라우터 등록
app.include_router(router)


if __name__ == "__main__":
    print("🤖 LucasAI Fine-tuning API 서버 시작...")
    print("📍 URL: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)