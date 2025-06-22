# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from datetime import datetime
import logging

from app.finefune.models import (
    TrainingResponse, TrainingJob,
    JobListResponse, HealthCheck, JobStatus
)
from .utils import (
    generate_job_id, validate_csv_file, check_system_resources,
    estimate_training_time, save_job_info, get_all_jobs,
    load_job_info
)
from app.finefune.finetune import start_finetune_job

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="🤖 LucasAI Fine-tuning API",
    description="CSV 업로드로 LucasAI 모델을 쉽게 파인튜닝하세요",
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

# 필요한 디렉토리 생성
for directory in ["uploads", "models", "logs", "jobs"]:
    Path(directory).mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지 - 간단한 업로드 인터페이스"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 LucasAI Fine-tuning API</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                margin: 20px 0;
            }
            .header {
                text-align: center;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
            }
            input, button { 
                padding: 12px; 
                margin: 8px 0; 
                border: 2px solid #e9ecef;
                border-radius: 8px; 
                font-size: 14px;
                width: 100%;
                box-sizing: border-box;
            }
            button { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                border: none; 
                cursor: pointer; 
                font-weight: bold;
            }
            button:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .job { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 4px solid #007bff;
            }
            .status-completed { border-left-color: #28a745; }
            .status-failed { border-left-color: #dc3545; }
            .status-training { border-left-color: #17a2b8; }
            .progress-bar {
                width: 100%;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 5px 0;
            }
            .progress-fill {
                height: 20px;
                background: linear-gradient(45deg, #28a745, #20c997);
                transition: width 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 LucasAI Fine-tuning API</h1>
                <p>CSV 파일을 업로드하면 자동으로 파인튜닝이 시작됩니다</p>
            </div>

            <form id="uploadForm" enctype="multipart/form-data">
                <h3>📁 새 모델 훈련</h3>
                <input type="text" id="modelName" placeholder="모델 이름 (예: lucasai-v1)" required>
                <input type="file" id="csvFile" accept=".csv" required>
                <button type="submit">🚀 파인튜닝 시작</button>
            </form>
        </div>

        <div class="container">
            <h3>📊 훈련 상태</h3>
            <div id="jobs"></div>
            <button onclick="refreshJobs()" type="button">🔄 새로고침</button>
        </div>

        <div class="container">
            <h3>📖 API 문서</h3>
            <p>🔗 <a href="/docs" target="_blank">Swagger UI</a></p>
            <p>🔗 <a href="/redoc" target="_blank">ReDoc</a></p>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const formData = new FormData();
                formData.append('csv_file', document.getElementById('csvFile').files[0]);
                formData.append('model_name', document.getElementById('modelName').value);

                try {
                    const response = await fetch('/train', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        alert('🎉 파인튜닝이 시작되었습니다!\\nJob ID: ' + result.job_id);
                        document.getElementById('uploadForm').reset();
                        refreshJobs();
                    } else {
                        alert('❌ 오류: ' + result.message);
                    }
                } catch (error) {
                    alert('❌ 오류: ' + error.message);
                }
            });

            async function refreshJobs() {
                try {
                    const response = await fetch('/jobs');
                    const data = await response.json();

                    const jobsDiv = document.getElementById('jobs');
                    jobsDiv.innerHTML = '';

                    if (data.jobs.length === 0) {
                        jobsDiv.innerHTML = '<p>📭 진행 중인 작업이 없습니다.</p>';
                        return;
                    }

                    data.jobs.forEach(job => {
                        const jobDiv = document.createElement('div');
                        jobDiv.className = `job status-${job.status}`;

                        const statusEmoji = {
                            'queued': '⏳',
                            'processing': '🔄',
                            'training': '🔥',
                            'completed': '✅',
                            'failed': '❌'
                        };

                        const progressBar = job.progress > 0 ? 
                            `<div class="progress-bar">
                                <div class="progress-fill" style="width: ${job.progress}%">
                                    ${job.progress}%
                                </div>
                             </div>` : '';

                        jobDiv.innerHTML = `
                            <h4>${statusEmoji[job.status]} ${job.model_name}</h4>
                            <p><strong>Job ID:</strong> ${job.job_id}</p>
                            <p><strong>상태:</strong> ${job.status}</p>
                            <p><strong>시작:</strong> ${new Date(job.created_at).toLocaleString()}</p>
                            ${job.message ? `<p><strong>메시지:</strong> ${job.message}</p>` : ''}
                            ${progressBar}
                            <button onclick="viewLogs('${job.job_id}')" type="button" style="width: auto; margin-right: 10px;">📋 로그</button>
                            ${job.status === 'completed' ? 
                                `<button onclick="downloadModel('${job.job_id}')" type="button" style="width: auto; margin-right: 10px;">📦 모델 다운로드</button>
                                 <button onclick="registerToOllama('${job.job_id}', '${job.model_name}')" type="button" style="width: auto; background: #28a745;">🚀 Ollama 등록</button>` : ''}
                        `;
                        jobsDiv.appendChild(jobDiv);
                    });
                } catch (error) {
                    console.error('Failed to refresh jobs:', error);
                }
            }

            async function viewLogs(jobId) {
                try {
                    const response = await fetch(`/jobs/${jobId}/logs`);
                    const text = await response.text();

                    const newWindow = window.open('', '_blank');
                    newWindow.document.write(`
                        <html>
                            <head><title>📋 로그 - ${jobId}</title></head>
                            <body style="font-family: monospace; white-space: pre-wrap; padding: 20px; background: #1e1e1e; color: #d4d4d4;">
                                <h2 style="color: #569cd6;">🤖 LucasAI 파인튜닝 로그 - ${jobId}</h2>
                                ${text.replace(/\\n/g, '<br>')}
                            </body>
                        </html>
                    `);
                } catch (error) {
                    alert('❌ 로그를 불러올 수 없습니다: ' + error.message);
                }
            }

            function downloadModel(jobId) {
                window.open(`/jobs/${jobId}/download`, '_blank');
            }

            async function registerToOllama(jobId, modelName) {
                if (!confirm(`'${modelName}' 모델을 Ollama에 등록하시겠습니까?`)) {
                    return;
                }

                try {
                    const response = await fetch(`/jobs/${jobId}/register-ollama`, {
                        method: 'POST'
                    });

                    const result = await response.json();

                    if (response.ok) {
                        alert(`✅ ${result.message}\\n\\n🌐 OpenWebUI에서 확인하세요: ${result.openwebui_url}`);
                        refreshJobs();
                    } else {
                        alert(`❌ 등록 실패: ${result.detail}`);
                    }
                } catch (error) {
                    alert('❌ 오류: ' + error.message);
                }
            }

            // 페이지 로드시 작업 목록 새로고침
            refreshJobs();

            // 10초마다 자동 새로고침
            setInterval(refreshJobs, 10000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스 체크"""
    resources = check_system_resources()
    return HealthCheck(
        status="healthy",
        gpu_available=resources["gpu_available"]
    )


@app.post("/train", response_model=TrainingResponse)
async def start_training(
        background_tasks: BackgroundTasks,
        model_name: str = Form(..., description="모델 이름"),
        csv_file: UploadFile = File(..., description="CSV 파일"),
        epochs: int = Form(2, description="훈련 에포크"),
        learning_rate: float = Form(2e-4, description="학습률")
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

    # 파일 저장
    upload_path = Path("uploads") / f"{job_id}_{csv_file.filename}"
    with open(upload_path, "wb") as f:
        f.write(file_content)

    # 훈련 설정
    config = {
        "base_model": "CarrotAI/Llama-3.2-Rabbit-Ko-1B-Instruct",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": 2,
        "max_length": 512,
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
        "csv_file": str(upload_path),
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


@app.get("/jobs", response_model=JobListResponse)
async def get_all_training_jobs():
    """모든 훈련 작업 조회"""
    jobs_data = get_all_jobs()

    jobs = []
    for job_data in jobs_data:
        job = TrainingJob(**job_data)
        jobs.append(job)

    return JobListResponse(jobs=jobs, total=len(jobs))


@app.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_job_status(job_id: str):
    """특정 작업 상태 조회"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return TrainingJob(**job_data)


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """작업 로그 조회"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    log_file = Path("logs") / f"{job_id}.log"

    if not log_file.exists():
        return "로그 파일이 아직 생성되지 않았습니다."

    with open(log_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/jobs/{job_id}/download")
async def download_model(job_id: str):
    """훈련된 모델 다운로드"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="완료된 작업만 다운로드 가능합니다")

    model_path = Path("models") / job_id

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")

    # Modelfile 반환
    modelfile_path = model_path / "Modelfile"
    if modelfile_path.exists():
        return FileResponse(
            path=modelfile_path,
            filename=f"{job_data['model_name']}_Modelfile",
            media_type="text/plain"
        )
    else:
        raise HTTPException(status_code=404, detail="Modelfile을 찾을 수 없습니다")


@app.post("/jobs/{job_id}/register-ollama")
async def register_model_to_ollama(job_id: str):
    """완료된 모델을 수동으로 Ollama에 등록"""
    job_data = load_job_info(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="완료된 작업만 등록할 수 있습니다")

    model_name = job_data["model_name"]
    model_dir = Path("models") / job_id
    modelfile_path = model_dir / "Modelfile"

    if not modelfile_path.exists():
        raise HTTPException(status_code=404, detail="Modelfile을 찾을 수 없습니다")

    try:
        import subprocess

        # 1. Modelfile을 ollama 컨테이너에 복사
        temp_modelfile = f"/tmp/Modelfile_{job_id}"
        copy_cmd = [
            "docker", "cp",
            str(modelfile_path),
            f"ollama:{temp_modelfile}"
        ]

        result = subprocess.run(copy_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"파일 복사 실패: {result.stderr}")

        # 2. Ollama에서 모델 생성
        create_cmd = [
            "docker", "exec", "ollama",
            "ollama", "create", model_name,
            "-f", temp_modelfile
        ]

        result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"모델 생성 실패: {result.stderr}")

        # 3. 등록 확인
        list_cmd = ["docker", "exec", "ollama", "ollama", "list"]
        result = subprocess.run(list_cmd, capture_output=True, text=True)

        if model_name not in result.stdout:
            raise HTTPException(status_code=500, detail="모델 등록 확인 실패")

        # 4. 작업 정보 업데이트
        update_job_status(job_id, "completed", ollama_registered=True)

        return {
            "message": f"'{model_name}' 모델이 Ollama에 성공적으로 등록되었습니다",
            "model_name": model_name,
            "openwebui_url": "http://localhost:3000"
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Ollama 모델 생성 시간 초과")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"등록 오류: {str(e)}")


@app.get("/ollama/models")
async def get_ollama_models():
    """Ollama에 등록된 모델 목록 조회"""
    try:
        import subprocess

        list_cmd = ["docker", "exec", "ollama", "ollama", "list"]
        result = subprocess.run(list_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Ollama 모델 목록 조회 실패")

        # 모델 목록 파싱
        lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
        models = []

        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    models.append({
                        "name": parts[0],
                        "id": parts[1] if len(parts) > 1 else "",
                        "size": parts[2] if len(parts) > 2 else "",
                        "modified": " ".join(parts[3:]) if len(parts) > 3 else ""
                    })

        return {"models": models}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류: {str(e)}")


@app.delete("/ollama/models/{model_name}")
async def delete_ollama_model(model_name: str):
    """Ollama에서 모델 삭제"""
    try:
        import subprocess

        delete_cmd = ["docker", "exec", "ollama", "ollama", "rm", model_name]
        result = subprocess.run(delete_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"모델 삭제 실패: {result.stderr}")

        return {"message": f"'{model_name}' 모델이 삭제되었습니다"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 오류: {str(e)}")


if __name__ == "__main__":
    print("🤖 LucasAI Fine-tuning API 서버 시작...")
    print("📍 URL: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)