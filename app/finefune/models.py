# app/models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    """훈련 작업 상태"""
    QUEUED = "queued"
    PROCESSING = "processing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingRequest(BaseModel):
    """훈련 요청 모델"""
    model_name: str = Field(..., description="생성할 모델 이름", example="lucasai-v1")
    base_model: Optional[str] = Field("microsoft/DialoGPT-medium", description="기본 모델")
    epochs: Optional[int] = Field(2, ge=1, le=10, description="훈련 에포크 수")
    learning_rate: Optional[float] = Field(2e-4, gt=0, description="학습률")

class TrainingJob(BaseModel):
    """훈련 작업 정보"""
    job_id: str = Field(..., description="작업 ID")
    model_name: str = Field(..., description="모델 이름")
    status: JobStatus = Field(JobStatus.QUEUED, description="작업 상태")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    started_at: Optional[datetime] = Field(None, description="시작 시간")
    completed_at: Optional[datetime] = Field(None, description="완료 시간")
    progress: int = Field(0, ge=0, le=100, description="진행률 (%)")
    message: Optional[str] = Field(None, description="상태 메시지")
    csv_file: Optional[str] = Field(None, description="CSV 파일 경로")
    model_path: Optional[str] = Field(None, description="모델 저장 경로")
    log_file: Optional[str] = Field(None, description="로그 파일 경로")

class TrainingResponse(BaseModel):
    """훈련 시작 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="초기 상태")
    message: str = Field(..., description="응답 메시지")
    estimated_time: Optional[str] = Field(None, description="예상 완료 시간")

class JobListResponse(BaseModel):
    """작업 목록 응답"""
    jobs: List[TrainingJob] = Field(..., description="훈련 작업 목록")
    total: int = Field(..., description="전체 작업 수")

class HealthCheck(BaseModel):
    """헬스 체크 응답"""
    status: str = Field("healthy", description="서비스 상태")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
    version: str = Field("1.0.0", description="API 버전")
    gpu_available: bool = Field(..., description="GPU 사용 가능 여부")

class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str = Field(..., description="에러 타입")
    message: str = Field(..., description="에러 메시지")
    timestamp: datetime = Field(default_factory=datetime.now, description="에러 발생 시간")