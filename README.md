# LucasAI Fine-tuning Pipeline


---
## 개요

이 프로젝트는 LucasAI의 AI 모델 파인튜닝을 자동화하는 웹 기반 플랫폼입니다. 사용자가 CSV 데이터를 업로드하면 자동으로 LoRA 파인튜닝을 수행하고, 완성된 모델을 Ollama에 등록하여 즉시 사용할 수 있도록 지원합니다.

---

## 프로젝트 구조

```
ollama_pipeline/
├── app/                          # 메인 애플리케이션
│   ├── main.py                   # FastAPI 애플리케이션 엔트리포인트
│   ├── api_routes.py             # API 라우터 및 엔드포인트
│   ├── templates.py              # HTML 템플릿 관리
│   ├── utils.py                  # 유틸리티 함수들
│   ├── path_manager.py           # 파일 경로 관리 클래스
│   ├── ollama_service.py         # Ollama 서비스 통합
│   └── finefune/                 # 파인튜닝 모듈
│       ├── finetune.py           # 메인 파인튜닝 로직
│       └── models.py             # Pydantic 모델 정의
├── workspace/                    # 작업 디렉토리
│   └── {job_id}/                 # 각 작업별 디렉토리
│       ├── data.csv              # 업로드된 훈련 데이터
│       ├── job_info.json         # 작업 메타데이터
│       ├── training.log          # 훈련 로그
│       └── model/                # 모델 파일들
│           ├── Modelfile         # Ollama Modelfile
│           ├── merged_model/     # LoRA 병합된 모델
│           ├── gguf/             # GGUF 변환된 모델
│           └── checkpoints/      # 체크포인트
├── Dockerfile                    # 컨테이너 빌드 설정
├── requirements.txt              # Python 의존성
└── README.md                     # README
```

---

## 설치 및 실행

### 1. 시스템 요구사항
- **OS**: Ubuntu 20.04+ (권장)
- **Python**: 3.10+
- **GPU**: CUDA 11.8+ (옵션, CPU도 지원)
- **RAM**: 8GB+ (권장 16GB+)
- **Storage**: 50GB+ (모델 저장용)
- **Docker**: 최신 버전

### 2. 의존성 설치
```bash
# 저장소 클론
git clone <repository-url>
cd ollama_pipeline

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
# 개발 모드
python -m app.main

# 또는 Docker 사용
docker build -t lucasai-finetune .
docker run -p 8000:8000 lucasai-finetune
```

### 4. 접속 확인
- **웹 인터페이스**: http://localhost:8000/finetune
- **API 문서**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 사용법

### 웹 인터페이스 사용
1. **http://localhost:8000/finetune** 접속
2. **모델 이름** 입력 (예: `lucasai-customer-v1`)
3. **CSV 파일** 업로드 (instruction, output 컬럼 필수)
4. **훈련 설정** 조정:
   - **베이스 모델**: CarrotAI Rabbit-Ko 1B (기본) 또는 커스텀
   - **에포크**: 2-3 (권장)
   - **학습률**: 0.0002 (기본)
   - **배치 크기**: 2 (기본)
   - **최대 길이**: 512 (기본)
5. **파인튜닝 시작** 클릭
6. **훈련 상태** 섹션에서 진행상황 모니터링

### CSV 데이터 형식
```csv
instruction,output
"LucasAI의 슬로건은 무엇인가요?","Making better choices with the power of AI입니다."
"Neuranex AI Platform의 특징을 알려주세요","AI Pipeline Builder, Hybrid Cloud, Integrated Management 등의 특징이 있습니다."
```

### API 직접 사용
```bash
# 파인튜닝 시작
curl -X POST "http://localhost:8000/train" \
  -F "csv_file=@data.csv" \
  -F "model_name=my-model" \
  -F "epochs=2" \
  -F "learning_rate=0.0002"

# 작업 상태 확인
curl "http://localhost:8000/jobs"

# 특정 작업 상세
curl "http://localhost:8000/jobs/{job_id}"

# 작업 삭제
curl -X DELETE "http://localhost:8000/jobs/{job_id}"
```

---

## 운영 가이드

### Ollama 및 Open WebUI 설치 (Docker)
참고 : https://www.youtube.com/watch?v=3rmoHTrUnk4
#### 1. Ollama Docker 설치
```bash
# Ollama 컨테이너 실행 (GPU 지원)
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama:latest

# CPU 전용 환경
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama:latest

# Ollama 서비스 상태 확인
curl http://localhost:11434/api/version
```

#### 2. Open WebUI Docker 설치
```bash
# Open WebUI 컨테이너 실행
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

# Linux 환경에서 Ollama와 연결
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -v open-webui:/app/backend/data \
  --link ollama \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

# 접속 확인
# Open WebUI: http://localhost:3000
# Ollama API: http://localhost:11434
```

#### 3. 기본 모델 다운로드
```bash
# Ollama 컨테이너 내부에서 모델 다운로드
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull qwen2.5:7b

# 모델 목록 확인
docker exec -it ollama ollama list
```

### 작업 모니터링
```bash
# 실시간 로그 확인
tail -f workspace/{job_id}/training.log

# 시스템 리소스 모니터링
htop
nvidia-smi  # GPU 사용 시

# 디스크 사용량 확인
du -sh workspace/

# Docker 컨테이너 상태 확인
docker ps
docker logs ollama
docker logs open-webui
```
