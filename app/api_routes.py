"""API 라우터 정의"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from datetime import datetime

from app.finefune.models import (
    TrainingResponse, TrainingJob,
    JobListResponse, HealthCheck, JobStatus
)
from .utils import (
    generate_job_id, validate_csv_file, check_system_resources,
    estimate_training_time, save_job_info, get_all_jobs,
    load_job_info, get_job_data_file, get_job_log_file,
    get_job_model_dir, get_job_modelfile, migrate_legacy_structure
)
from app.finefune.finetune import start_finetune_job
from .templates import get_main_page_template
from .ollama_service import register_model_to_ollama, get_ollama_models, delete_ollama_model


# 라우터 생성
router = APIRouter()


@router.get("/finetune", response_class=HTMLResponse)
async def root():
    """메인 페이지 - 간단한 업로드 인터페이스"""
    return HTMLResponse(content=get_main_page_template())


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스 체크"""
    resources = check_system_resources()
    return HealthCheck(
        status="healthy",
        gpu_available=resources["gpu_available"]
    )


@router.post("/train", response_model=TrainingResponse)
async def start_training(
        background_tasks: BackgroundTasks,
        model_name: str = Form(..., description="모델 이름"),
        csv_file: UploadFile = File(..., description="CSV 파일"),
        base_model: str = Form("CarrotAI/Llama-3.2-Rabbit-Ko-1B-Instruct", description="베이스 모델"),
        epochs: int = Form(2, description="훈련 에포크"),
        learning_rate: float = Form(2e-4, description="학습률"),
        batch_size: int = Form(2, description="배치 크기"),
        max_length: int = Form(512, description="최대 시퀀스 길이")
):
    """CSV 파일 업로드 및 파인튜닝 시작"""

    # 파일 확장자 확인
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다")

    # CSV 파일 검증
    file_content = await csv_file.read()
    is_valid, message, df = validate_csv_file(file_content)

    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    # Job ID 생성
    job_id = generate_job_id()

    # 파일 저장 (새로운 경로 구조 사용)
    upload_path = get_job_data_file(job_id)
    with open(upload_path, "wb") as f:
        f.write(file_content)

    # 훈련 설정 (사용자 입력값 사용)
    config = {
        "base_model": base_model,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1
    }

    # 작업 정보 저장
    job_data = {
        "job_id": job_id,
        "model_name": model_name,
        "status": JobStatus.QUEUED.value,
        "created_at": datetime.now().isoformat(),
        "data_file": str(upload_path),
        "config": config,
        "progress": 0,
        "message": "작업이 대기열에 추가되었습니다"
    }

    save_job_info(job_data, job_id)

    # 백그라운드에서 파인튜닝 시작
    background_tasks.add_task(
        start_finetune_job,
        job_id,
        str(upload_path),
        model_name,
        config
    )

    # 예상 시간 계산
    estimated_time = estimate_training_time(len(df), epochs)

    return TrainingResponse(
        job_id=job_id,
        status="queued",
        message=f"파인튜닝이 시작됩니다. 데이터: {len(df)}개",
        estimated_time=estimated_time
    )


@router.get("/jobs", response_model=JobListResponse)
async def get_all_training_jobs():
    """모든 훈련 작업 조회"""
    jobs_data = get_all_jobs()

    jobs = []
    for job_data in jobs_data:
        job = TrainingJob(**job_data)
        jobs.append(job)

    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_job_status(job_id: str):
    """특정 작업 상태 조회"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return TrainingJob(**job_data)


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """작업 로그 조회"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    log_file = get_job_log_file(job_id)

    if not log_file.exists():
        return "로그 파일이 아직 생성되지 않았습니다."

    with open(log_file, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/jobs/{job_id}/download")
async def download_model(job_id: str):
    """훈련된 모델 다운로드"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="완료된 작업만 다운로드 가능합니다")

    model_path = get_job_model_dir(job_id)

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")

    # Modelfile 반환
    modelfile_path = get_job_modelfile(job_id)
    if modelfile_path.exists():
        return FileResponse(
            path=modelfile_path,
            filename=f"{job_data['model_name']}_Modelfile",
            media_type="text/plain"
        )
    else:
        raise HTTPException(status_code=404, detail="Modelfile을 찾을 수 없습니다")


@router.post("/jobs/{job_id}/register-ollama")
async def register_model_endpoint(job_id: str):
    """완료된 모델을 수동으로 Ollama에 등록"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return await register_model_to_ollama(job_id, job_data)


@router.get("/ollama/models")
async def get_ollama_models_endpoint():
    """Ollama에 등록된 모델 목록 조회"""
    return await get_ollama_models()


@router.delete("/ollama/models/{model_name}")
async def delete_ollama_model_endpoint(model_name: str):
    """Ollama에서 모델 삭제"""
    return await delete_ollama_model(model_name)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """작업 삭제 (파일 및 디렉토리 완전 삭제)"""
    from .path_manager import path_manager
    
    job_data = load_job_info(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    try:
        # workspace에서 작업 디렉토리 전체 삭제
        if path_manager.cleanup_job(job_id):
            return {"message": f"작업 '{job_id}'가 성공적으로 삭제되었습니다"}
        else:
            raise HTTPException(status_code=500, detail="작업 삭제에 실패했습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 오류: {str(e)}")