"""HTML 템플릿 관리"""

def get_main_page_template() -> str:
    """메인 페이지 HTML 템플릿 반환"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LucasAI Fine-tuning API</title>
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
            input, button, select { 
                padding: 12px; 
                margin: 8px 0; 
                border: 2px solid #e9ecef;
                border-radius: 8px; 
                font-size: 14px;
                width: 100%;
                box-sizing: border-box;
            }
            label {
                font-weight: bold;
                margin: 8px 0 4px 0;
                display: block;
                color: #495057;
            }
            small {
                display: block;
                margin-top: 2px;
                font-size: 12px;
            }
            h4 {
                margin: 20px 0 10px 0;
                color: #495057;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 5px;
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
                <h1>LucasAI Fine-tuning API</h1>
                <p>CSV 파일을 업로드하면 자동으로 파인튜닝이 시작됩니다</p>
            </div>

            <form id="uploadForm" enctype="multipart/form-data">
                <h3>📁 새 모델 훈련</h3>
                
                <label for="modelName">🏷️ 모델 이름:</label>
                <input type="text" id="modelName" placeholder="모델 이름 (예: lucasai-v1)" required>
                
                <label for="csvFile">📄 데이터 파일:</label>
                <input type="file" id="csvFile" accept=".csv" required>
                
                <h4>⚙️ 훈련 설정</h4>
                
                <div style="margin-bottom: 15px;">
                    <label for="baseModel">🤖 베이스 모델:</label>
                    <select id="baseModel" required>
                        <option value="CarrotAI/Llama-3.2-Rabbit-Ko-1B-Instruct" selected>CarrotAI/Llama-3.2-Rabbit-Ko-1B-Instruct</option>
                        <option value="custom">🔧 직접 입력</option>
                    </select>
                    <input type="text" id="customModel" placeholder="예: huggingface-model-name" style="display: none; margin-top: 8px;">
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <label for="epochs">🔄 에포크 (Epochs):</label>
                        <input type="number" id="epochs" min="1" max="10" value="2" required>
                        <small style="color: #666;">권장: 2-3</small>
                    </div>
                    
                    <div>
                        <label for="learningRate">📈 학습률 (Learning Rate):</label>
                        <select id="learningRate" required>
                            <option value="1e-4">0.0001 (안전)</option>
                            <option value="2e-4" selected>0.0002 (기본)</option>
                            <option value="3e-4">0.0003 (빠름)</option>
                            <option value="5e-4">0.0005 (공격적)</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="batchSize">📦 배치 크기:</label>
                        <select id="batchSize" required>
                            <option value="1">1 (메모리 절약)</option>
                            <option value="2" selected>2 (기본)</option>
                            <option value="4">4 (빠름)</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="maxLength">📏 최대 길이:</label>
                        <select id="maxLength" required>
                            <option value="256">256 (짧음)</option>
                            <option value="512" selected>512 (기본)</option>
                            <option value="1024">1024 (긴 텍스트)</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" style="margin-top: 15px;">🚀 파인튜닝 시작</button>
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
            // 베이스 모델 선택 시 커스텀 입력 필드 토글
            document.getElementById('baseModel').addEventListener('change', function() {
                const customInput = document.getElementById('customModel');
                if (this.value === 'custom') {
                    customInput.style.display = 'block';
                    customInput.required = true;
                } else {
                    customInput.style.display = 'none';
                    customInput.required = false;
                    customInput.value = '';
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                // 베이스 모델 값 결정
                const baseModelSelect = document.getElementById('baseModel');
                const customModelInput = document.getElementById('customModel');
                const baseModelValue = baseModelSelect.value === 'custom' ? customModelInput.value : baseModelSelect.value;

                const formData = new FormData();
                formData.append('csv_file', document.getElementById('csvFile').files[0]);
                formData.append('model_name', document.getElementById('modelName').value);
                formData.append('base_model', baseModelValue);
                formData.append('epochs', document.getElementById('epochs').value);
                formData.append('learning_rate', document.getElementById('learningRate').value);
                formData.append('batch_size', document.getElementById('batchSize').value);
                formData.append('max_length', document.getElementById('maxLength').value);

                try {
                    const response = await fetch('/train', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        alert('🎉 파인튜닝이 시작되었습니다!\\nJob ID: ' + result.job_id + '\\n\\n⚙️ 설정:\\n' +
                              '- 베이스 모델: ' + baseModelValue + '\\n' +
                              '- 에포크: ' + document.getElementById('epochs').value + '\\n' +
                              '- 학습률: ' + document.getElementById('learningRate').value + '\\n' +
                              '- 배치 크기: ' + document.getElementById('batchSize').value + '\\n' +
                              '- 최대 길이: ' + document.getElementById('maxLength').value);
                        document.getElementById('uploadForm').reset();
                        // 기본값으로 다시 설정
                        document.getElementById('baseModel').value = 'CarrotAI/Llama-3.2-Rabbit-Ko-1B-Instruct';
                        document.getElementById('customModel').style.display = 'none';
                        document.getElementById('customModel').required = false;
                        document.getElementById('epochs').value = '2';
                        document.getElementById('learningRate').value = '2e-4';
                        document.getElementById('batchSize').value = '2';
                        document.getElementById('maxLength').value = '512';
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
                                `<button onclick="downloadModel('${job.job_id}')" type="button" style="width: auto; margin-right: 10px;">📦 모델 다운로드</button>` : ''}
                            <button onclick="deleteJob('${job.job_id}', '${job.model_name}')" type="button" style="width: auto; background: #dc3545; margin-left: 10px;">🗑️ 삭제</button>
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

            async function deleteJob(jobId, modelName) {
                if (!confirm(`정말로 '${modelName}' 작업을 삭제하시겠습니까?\\n\\n⚠️ 이 작업은 되돌릴 수 없으며, 모든 관련 파일(로그, 모델, 데이터)이 영구적으로 삭제됩니다.`)) {
                    return;
                }

                try {
                    const response = await fetch(`/jobs/${jobId}`, {
                        method: 'DELETE'
                    });

                    const result = await response.json();

                    if (response.ok) {
                        alert(`✅ ${result.message}`);
                        refreshJobs(); // 목록 새로고침
                    } else {
                        alert(`❌ 삭제 실패: ${result.detail}`);
                    }
                } catch (error) {
                    alert('❌ 삭제 중 오류가 발생했습니다: ' + error.message);
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