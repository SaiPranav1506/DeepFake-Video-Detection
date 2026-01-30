let metricsData = {};

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    if (tabName === 'training') {
        loadMetrics();
    } else if (tabName === 'model') {
        loadModelInfo();
    }
}

async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        metricsData = await response.json();
        
        renderMetricsChart();
        renderConfusionMatrices();
        renderMetricsTable();
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

function renderMetricsChart() {
    const epochs = Object.keys(metricsData).map(e => parseInt(e)).sort((a, b) => a - b);
    
    if (epochs.length === 0) {
        document.getElementById('metrics-chart').innerHTML = '<p style="text-align: center; color: #94a3b8;">No training data available yet</p>';
        return;
    }

    const accuracy = epochs.map(e => metricsData[e].accuracy * 100);
    const precision = epochs.map(e => metricsData[e].precision * 100);
    const recall = epochs.map(e => metricsData[e].recall * 100);
    const f1 = epochs.map(e => metricsData[e].f1 * 100);
    const auc = epochs.map(e => metricsData[e].auc ? metricsData[e].auc * 100 : null);

    const trace1 = {
        x: epochs,
        y: accuracy,
        mode: 'lines+markers',
        name: 'Accuracy',
        line: { color: '#2563eb', width: 3 },
        marker: { size: 8 }
    };

    const trace2 = {
        x: epochs,
        y: precision,
        mode: 'lines+markers',
        name: 'Precision',
        line: { color: '#7c3aed', width: 3 },
        marker: { size: 8 }
    };

    const trace3 = {
        x: epochs,
        y: recall,
        mode: 'lines+markers',
        name: 'Recall',
        line: { color: '#10b981', width: 3 },
        marker: { size: 8 }
    };

    const trace4 = {
        x: epochs,
        y: f1,
        mode: 'lines+markers',
        name: 'F1 Score',
        line: { color: '#f59e0b', width: 3 },
        marker: { size: 8 }
    };

    const traces = [trace1, trace2, trace3, trace4];
    
    if (auc.some(v => v !== null)) {
        const trace5 = {
            x: epochs,
            y: auc,
            mode: 'lines+markers',
            name: 'AUC',
            line: { color: '#ef4444', width: 3 },
            marker: { size: 8 }
        };
        traces.push(trace5);
    }

    const layout = {
        title: 'Training Metrics Over Epochs',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Score (%)' },
        hovermode: 'x unified',
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0' },
        margin: { l: 60, r: 40, t: 60, b: 40 }
    };

    Plotly.newPlot('metrics-chart', traces, layout, { responsive: true });
}

function renderConfusionMatrices() {
    const epochs = Object.keys(metricsData).map(e => parseInt(e)).sort((a, b) => a - b);
    const container = document.getElementById('confusion-grid');
    container.innerHTML = '';

    epochs.forEach(epoch => {
        const cm = metricsData[epoch].confusion_matrix;
        const div = document.createElement('div');
        div.className = 'confusion-matrix';
        
        let html = `<h3>Epoch ${epoch}</h3>`;
        html += '<table>';
        html += '<tr><td class="header"></td><td class="header">Pred Real</td><td class="header">Pred Fake</td></tr>';
        html += `<tr><td class="header actual">Actual Real</td><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>`;
        html += `<tr><td class="header actual">Actual Fake</td><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>`;
        html += '</table>';
        
        div.innerHTML = html;
        container.appendChild(div);
    });
}

function renderMetricsTable() {
    const epochs = Object.keys(metricsData).map(e => parseInt(e)).sort((a, b) => a - b);
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = '';

    epochs.forEach(epoch => {
        const m = metricsData[epoch];
        const row = tbody.insertRow();
        
        row.innerHTML = `
            <td>${epoch}</td>
            <td>${(m.accuracy * 100).toFixed(2)}%</td>
            <td>${(m.precision * 100).toFixed(2)}%</td>
            <td>${(m.recall * 100).toFixed(2)}%</td>
            <td>${(m.f1 * 100).toFixed(2)}%</td>
            <td>${m.auc ? (m.auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            <td>${m.total_samples}</td>
        `;
    });
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const info = await response.json();
        
        const container = document.getElementById('model-info');
        let html = '';
        
        if (info.loaded) {
            html += '<div class="info-item">';
            html += '<span class="label">Status:</span> ✅ Model Loaded';
            html += '</div>';
            html += '<div class="info-item">';
            html += `<span class="label">Checkpoint:</span> ${info.checkpoint}`;
            html += '</div>';
            html += '<div class="info-item">';
            html += `<span class="label">Model Type:</span> ${info.model_type}`;
            html += '</div>';
        } else {
            html += '<div class="info-item">';
            html += '<span class="label">Status:</span> ⚠️ No Model Loaded';
            html += '</div>';
        }
        
        html += '<div class="info-item">';
        html += `<span class="label">Device:</span> ${info.device}`;
        html += '</div>';
        
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

function loadModel() {
    const checkpoint = document.getElementById('checkpoint-select').value;
    const modelType = document.getElementById('model-type-select').value;
    
    if (!checkpoint) {
        updateLoadStatus('Please select a checkpoint', 'error');
        return;
    }

    const btn = event.target;
    btn.disabled = true;
    updateLoadStatus('Loading...', '');

    fetch('/api/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint, model_type: modelType })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            updateLoadStatus('✅ Model loaded successfully!', 'success');
            loadModelInfo();
        } else {
            updateLoadStatus('❌ Error: ' + data.error, 'error');
        }
        btn.disabled = false;
    })
    .catch(e => {
        updateLoadStatus('❌ Error: ' + e.message, 'error');
        btn.disabled = false;
    });
}

function updateLoadStatus(message, className) {
    const status = document.getElementById('load-status');
    status.textContent = message;
    status.className = className;
}

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#2563eb';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        performPrediction();
    }
});

fileInput.addEventListener('change', performPrediction);

function performPrediction() {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const progressDiv = document.getElementById('upload-progress');
    progressDiv.style.display = 'block';

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(result => {
        progressDiv.style.display = 'none';
        displayResult(result);
    })
    .catch(error => {
        progressDiv.style.display = 'none';
        displayResult({ error: error.message });
    });
}

function displayResult(result) {
    const resultSection = document.getElementById('result-section');
    const resultBox = document.getElementById('result-box');
    
    if (result.error) {
        resultBox.innerHTML = `<p style="color: #ef4444;">❌ Error: ${result.error}</p>`;
        resultSection.style.display = 'block';
        return;
    }

    const isDeepfake = result.prediction === 'Deepfake';
    const confidencePercent = (result.confidence * 100).toFixed(1);
    const probRealPercent = (result.prob_real * 100).toFixed(1);
    const probFakePercent = (result.prob_fake * 100).toFixed(1);

    let html = `
        <div class="${isDeepfake ? 'deepfake' : 'real'}">
            <h3>${isDeepfake ? '⚠️ DEEPFAKE DETECTED' : '✅ REAL VIDEO'}</h3>
            <div class="prediction">${result.prediction}</div>
            <div class="confidence">Confidence: ${confidencePercent}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>
            <div style="margin-top: 20px; text-align: left; display: inline-block;">
                <p>📊 Prediction Breakdown:</p>
                <p>• Real: ${probRealPercent}%</p>
                <p>• Fake: ${probFakePercent}%</p>
            </div>
        </div>
    `;
    
    resultBox.innerHTML = html;
    resultSection.style.display = 'block';
}

window.addEventListener('load', () => {
    loadMetrics();
    loadModelInfo();
});
