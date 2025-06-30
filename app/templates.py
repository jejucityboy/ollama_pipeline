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