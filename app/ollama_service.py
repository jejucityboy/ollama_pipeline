"""Ollama 서비스 관리"""

import subprocess
from typing import Dict, List, Any
from pathlib import Path
from fastapi import HTTPException


async def register_model_to_ollama(job_id: str, job_data: Dict[str, Any]) -> Dict[str, str]:
    """완료된 모델을 Ollama에 등록"""
    if job_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="완료된 작업만 등록할 수 있습니다")

    from .utils import get_job_modelfile
    
    model_name = job_data["model_name"]
    modelfile_path = get_job_modelfile(job_id)

    if not modelfile_path.exists():
        raise HTTPException(status_code=404, detail="Modelfile을 찾을 수 없습니다")

    try:
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

        return {
            "message": f"'{model_name}' 모델이 Ollama에 성공적으로 등록되었습니다",
            "model_name": model_name,
            "openwebui_url": "http://localhost:3000"
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Ollama 모델 생성 시간 초과")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"등록 오류: {str(e)}")


async def get_ollama_models() -> Dict[str, List[Dict[str, str]]]:
    """Ollama에 등록된 모델 목록 조회"""
    try:
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


async def delete_ollama_model(model_name: str) -> Dict[str, str]:
    """Ollama에서 모델 삭제"""
    try:
        delete_cmd = ["docker", "exec", "ollama", "ollama", "rm", model_name]
        result = subprocess.run(delete_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"모델 삭제 실패: {result.stderr}")

        return {"message": f"'{model_name}' 모델이 삭제되었습니다"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 오류: {str(e)}")