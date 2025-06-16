# app/utils.py
import pandas as pd
import uuid
import torch
from pathlib import Path
from typing import Tuple, Dict, Any
import logging
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """고유한 작업 ID 생성"""
    return str(uuid.uuid4())[:8]


def validate_csv_file(file_content: bytes) -> Tuple[bool, str, pd.DataFrame]:
    """
    CSV 파일 유효성 검사

    Returns:
        Tuple[bool, str, pd.DataFrame]: (성공여부, 메시지, 데이터프레임)
    """
    try:
        # CSV 읽기
        df = pd.read_csv(pd.io.common.StringIO(file_content.decode('utf-8')))

        # 필수 컬럼 확인
        required_columns = ['instruction', 'output']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"필수 컬럼이 없습니다: {missing_columns}", None

        # 빈 데이터 확인
        if len(df) == 0:
            return False, "CSV 파일이 비어있습니다", None

        # 유효한 행 확인
        valid_df = df.dropna(subset=['instruction', 'output'])
        valid_df = valid_df[
            (valid_df['instruction'].str.strip() != '') &
            (valid_df['output'].str.strip() != '')
            ]

        if len(valid_df) < 5:
            return False, f"유효한 데이터가 너무 적습니다 ({len(valid_df)}개). 최소 5개 이상 필요합니다.", None

        return True, f"유효한 데이터 {len(valid_df)}개 확인", valid_df

    except Exception as e:
        return False, f"CSV 파일 처리 오류: {str(e)}", None


def check_system_resources() -> Dict[str, Any]:
    """시스템 리소스 확인"""
    resources = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": torch.get_num_threads(),
    }

    if resources["gpu_available"]:
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            resources["gpu_name"] = gpu_props.name
            resources["gpu_memory"] = gpu_props.total_memory // (1024 ** 3)  # GB
        except:
            pass

    return resources


def estimate_training_time(num_samples: int, epochs: int = 2) -> str:
    """훈련 시간 추정"""
    # 간단한 추정 공식 (실제로는 더 복잡함)
    base_time = 5  # 기본 5분
    sample_factor = num_samples / 100  # 100개당 추가 시간
    epoch_factor = epochs

    estimated_minutes = base_time + (sample_factor * epoch_factor)

    if estimated_minutes < 60:
        return f"약 {int(estimated_minutes)}분"
    else:
        hours = int(estimated_minutes // 60)
        minutes = int(estimated_minutes % 60)
        return f"약 {hours}시간 {minutes}분"


def save_job_info(job_data: Dict[str, Any], job_id: str) -> None:
    """작업 정보를 파일에 저장"""
    jobs_dir = Path("jobs")
    jobs_dir.mkdir(exist_ok=True)

    job_file = jobs_dir / f"{job_id}.json"
    with open(job_file, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, indent=2, ensure_ascii=False, default=str)


def load_job_info(job_id: str) -> Dict[str, Any]:
    """작업 정보 로드"""
    job_file = Path("jobs") / f"{job_id}.json"

    if not job_file.exists():
        return None

    with open(job_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_jobs() -> list:
    """모든 작업 정보 조회"""
    jobs_dir = Path("jobs")
    if not jobs_dir.exists():
        return []

    jobs = []
    for job_file in jobs_dir.glob("*.json"):
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
                jobs.append(job_data)
        except Exception as e:
            logger.error(f"Failed to load job file {job_file}: {e}")

    # 생성 시간순으로 정렬
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jobs


def update_job_status(job_id: str, status: str, **kwargs) -> None:
    """작업 상태 업데이트"""
    job_data = load_job_info(job_id)
    if not job_data:
        return

    job_data['status'] = status
    job_data['updated_at'] = datetime.now().isoformat()

    # 추가 필드 업데이트
    for key, value in kwargs.items():
        job_data[key] = value

    save_job_info(job_data, job_id)


def cleanup_old_files(days: int = 7) -> None:
    """오래된 파일 정리"""
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=days)

    for directory in ["uploads", "models", "logs", "jobs"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        for file_path in dir_path.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up {file_path}: {e}")


def format_file_size(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 형태로 변환"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """모델 정보 조회"""
    if not model_path.exists():
        return None

    info = {
        "path": str(model_path),
        "size": format_file_size(sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())),
        "created": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(),
        "files": [f.name for f in model_path.iterdir() if f.is_file()]
    }

    # model_info.json 파일이 있으면 추가 정보 로드
    info_file = model_path / "model_info.json"
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                info.update(model_info)
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")

    return info