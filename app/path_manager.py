"""파일 경로 관리"""

from pathlib import Path
from typing import Dict, Any
import shutil
import logging

logger = logging.getLogger(__name__)


class JobPathManager:
    """작업별 파일 경로 관리 클래스"""
    
    def __init__(self, workspace_root: str = "workspace"):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(exist_ok=True)
    
    def get_job_dir(self, job_id: str) -> Path:
        """작업 디렉토리 경로 반환"""
        job_dir = self.workspace_root / job_id
        job_dir.mkdir(exist_ok=True)
        return job_dir
    
    def get_data_file(self, job_id: str) -> Path:
        """업로드된 데이터 파일 경로"""
        return self.get_job_dir(job_id) / "data.csv"
    
    def get_job_info_file(self, job_id: str) -> Path:
        """작업 정보 파일 경로"""
        return self.get_job_dir(job_id) / "job_info.json"
    
    def get_log_file(self, job_id: str) -> Path:
        """로그 파일 경로"""
        return self.get_job_dir(job_id) / "training.log"
    
    def get_model_dir(self, job_id: str) -> Path:
        """모델 디렉토리 경로"""
        model_dir = self.get_job_dir(job_id) / "model"
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    def get_modelfile(self, job_id: str) -> Path:
        """Modelfile 경로"""
        return self.get_model_dir(job_id) / "Modelfile"
    
    def get_checkpoint_dir(self, job_id: str) -> Path:
        """체크포인트 디렉토리 경로"""
        checkpoint_dir = self.get_model_dir(job_id) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir
    
    def get_merged_model_dir(self, job_id: str) -> Path:
        """병합된 모델 디렉토리 경로"""
        merged_dir = self.get_model_dir(job_id) / "merged_model"
        merged_dir.mkdir(exist_ok=True)
        return merged_dir
    
    def get_gguf_dir(self, job_id: str) -> Path:
        """GGUF 파일 디렉토리 경로"""
        gguf_dir = self.get_model_dir(job_id) / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        return gguf_dir
    
    def cleanup_job(self, job_id: str) -> bool:
        """작업 디렉토리 전체 삭제"""
        try:
            job_dir = self.get_job_dir(job_id)
            if job_dir.exists():
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up job directory: {job_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            return False
    
    def get_job_size(self, job_id: str) -> int:
        """작업 디렉토리 총 크기 (바이트)"""
        job_dir = self.get_job_dir(job_id)
        if not job_dir.exists():
            return 0
        
        total_size = 0
        for file_path in job_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def list_all_jobs(self) -> list:
        """모든 작업 ID 목록 반환"""
        if not self.workspace_root.exists():
            return []
        
        jobs = []
        for job_dir in self.workspace_root.iterdir():
            if job_dir.is_dir():
                jobs.append(job_dir.name)
        
        return sorted(jobs, reverse=True)
    
    def migrate_legacy_files(self, job_id: str) -> bool:
        """기존 파일 구조를 새 구조로 마이그레이션"""
        try:
            job_dir = self.get_job_dir(job_id)
            
            # 기존 uploads 파일 이동
            uploads_dir = Path("uploads")
            if uploads_dir.exists():
                for file_path in uploads_dir.glob(f"{job_id}_*"):
                    target_path = self.get_data_file(job_id)
                    shutil.move(str(file_path), str(target_path))
                    logger.info(f"Migrated upload file: {file_path} -> {target_path}")
            
            # 기존 jobs 파일 이동
            jobs_dir = Path("jobs")
            job_info_file = jobs_dir / f"{job_id}.json"
            if job_info_file.exists():
                target_path = self.get_job_info_file(job_id)
                shutil.move(str(job_info_file), str(target_path))
                logger.info(f"Migrated job info: {job_info_file} -> {target_path}")
            
            # 기존 logs 파일 이동
            logs_dir = Path("logs")
            log_file = logs_dir / f"{job_id}.log"
            if log_file.exists():
                target_path = self.get_log_file(job_id)
                shutil.move(str(log_file), str(target_path))
                logger.info(f"Migrated log file: {log_file} -> {target_path}")
            
            # 기존 models 디렉토리 이동
            models_dir = Path("models")
            model_dir = models_dir / job_id
            if model_dir.exists():
                target_dir = self.get_model_dir(job_id)
                # model 디렉토리 내용을 복사
                for item in model_dir.iterdir():
                    target_item = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(str(item), str(target_item), dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(item), str(target_item))
                
                # 기존 디렉토리 삭제
                shutil.rmtree(str(model_dir))
                logger.info(f"Migrated model directory: {model_dir} -> {target_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate files for job {job_id}: {e}")
            return False


# 전역 인스턴스
path_manager = JobPathManager()