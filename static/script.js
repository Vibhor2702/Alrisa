// Global state
let currentData = null;
let currentFilename = null;

// Model options
const modelOptions = {
    'Classification': ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression'],
    'Regression': ['Decision Tree Regressor', 'Random Forest Regressor', 'Linear Regression', 'SVR'],
    'Clustering': ['K-Means', 'DBSCAN'],
    'Dimensionality Reduction': ['PCA', 't-SNE']
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateModelOptions();
    setupEventListeners();
});

function setupEventListeners() {
    // File upload
    document.getElementById('fileUpload').addEventListener('change', handleFileUpload);
    
    // Task selection
    document.getElementById('taskSelect').addEventListener('change', () => {
        updateModelOptions();
        updateConfigVisibility();
    });
    
    // Run analysis
    document.getElementById('runAnalysis').addEventListener('click', runAnalysis);
    
    // Help modal
    const modal = document.getElementById('chatModal');
    document.getElementById('helpToggle').addEventListener('click', () => {
        modal.classList.add('active');
    });
    document.querySelector('.close').addEventListener('click', () => {
        modal.classList.remove('active');
    });
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });
    
    // Chat
    document.getElementById('chatSend').addEventListener('click', sendChatMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
    
    // Download code
    document.getElementById('downloadCode').addEventListener('click', downloadCode);
}

function updateModelOptions() {
    const task = document.getElementById('taskSelect').value;
    const modelSelect = document.getElementById('modelSelect');
    modelSelect.innerHTML = '';
    
    modelOptions[task].forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
}

function updateConfigVisibility() {
    const task = document.getElementById('taskSelect').value;
    const targetSection = document.getElementById('targetSection');
    const featuresSection = document.getElementById('featuresSection');
    
    if (task === 'Classification' || task === 'Regression') {
        targetSection.style.display = 'block';
        featuresSection.style.display = 'block';
    } else {
        targetSection.style.display = 'none';
        featuresSection.style.display = 'block';
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Server returned invalid response. Please refresh the page and try again.');
        }
        
        const data = await response.json();
        
        if (data.success) {
            currentFilename = data.filename;
            currentData = data.eda;
            
            // Show filename
            const fileNameDiv = document.getElementById('fileName');
            fileNameDiv.textContent = `✓ ${file.name}`;
            fileNameDiv.classList.add('active');
            
            // Display EDA
            displayEDA(data.eda);
            
            // Populate columns
            populateColumns(data.eda.columns);
            
            // Enable run button
            document.getElementById('runAnalysis').disabled = false;
            
            // Hide results section if visible
            document.getElementById('resultsSection').style.display = 'none';
            
            // Hide welcome, show EDA
            document.getElementById('welcomeScreen').style.display = 'none';
            document.getElementById('edaSection').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading file: ' + error.message);
    }
    
    // Reset file input to allow uploading the same file again
    event.target.value = '';
}

function displayEDA(eda) {
    // Stats
    document.getElementById('rowCount').textContent = eda.shape[0];
    document.getElementById('colCount').textContent = eda.shape[1];
    
    const missingTotal = Object.values(eda.missing_values).reduce((a, b) => a + b, 0);
    document.getElementById('missingCount').textContent = missingTotal;
    document.getElementById('numericCount').textContent = eda.numeric_columns.length;
    
    // Data table
    if (eda.preview && eda.preview.length > 0) {
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        
        // Headers
        const headerRow = document.createElement('tr');
        Object.keys(eda.preview[0]).forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        
        // Rows
        eda.preview.forEach(row => {
            const tr = document.createElement('tr');
            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                td.textContent = value !== null ? value : 'N/A';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        
        table.appendChild(thead);
        table.appendChild(tbody);
        document.getElementById('dataTable').innerHTML = '';
        document.getElementById('dataTable').appendChild(table);
    }
    
    // Correlation matrix (simplified display)
    if (eda.correlation) {
        const correlationSection = document.getElementById('correlationSection');
        correlationSection.style.display = 'block';
        
        const corrDiv = document.getElementById('correlationMatrix');
        corrDiv.innerHTML = '<p style="color: var(--text-secondary);">Correlation data computed successfully. Numeric columns are correlated.</p>';
    }
}

function populateColumns(columns) {
    const targetSelect = document.getElementById('targetSelect');
    const featuresList = document.getElementById('featuresList');
    
    targetSelect.innerHTML = '';
    featuresList.innerHTML = '';
    
    columns.forEach(col => {
        // Target dropdown
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        targetSelect.appendChild(option);
        
        // Features checkboxes
        const div = document.createElement('div');
        div.className = 'feature-checkbox';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `feature_${col}`;
        checkbox.value = col;
        checkbox.checked = true;
        
        const label = document.createElement('label');
        label.htmlFor = `feature_${col}`;
        label.textContent = col;
        
        div.appendChild(checkbox);
        div.appendChild(label);
        featuresList.appendChild(div);
    });
    
    updateConfigVisibility();
}

async function runAnalysis() {
    const task = document.getElementById('taskSelect').value;
    const model = document.getElementById('modelSelect').value;
    const target = document.getElementById('targetSelect').value;
    
    // Get selected features
    const featureCheckboxes = document.querySelectorAll('#featuresList input[type="checkbox"]:checked');
    const features = Array.from(featureCheckboxes).map(cb => cb.value);
    
    if (features.length === 0) {
        alert('Please select at least one feature column.');
        return;
    }
    
    // Remove target from features if present
    const finalFeatures = features.filter(f => f !== target);
    
    if (task !== 'Clustering' && task !== 'Dimensionality Reduction' && finalFeatures.length === 0) {
        alert('Please select feature columns different from the target.');
        return;
    }
    
    // Show loading
    document.getElementById('loadingScreen').style.display = 'flex';
    document.getElementById('resultsSection').style.display = 'none';
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentFilename,
                task: task,
                model: model,
                target: target,
                features: task === 'Clustering' || task === 'Dimensionality Reduction' ? features : finalFeatures
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data, task);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error training model: ' + error.message);
    } finally {
        document.getElementById('loadingScreen').style.display = 'none';
    }
}

function displayResults(data, task) {
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = '';
    
    if (task === 'Classification') {
        resultsContent.innerHTML = `
            <div class="result-metric">
                <span class="result-label">Accuracy</span>
                <span class="result-value">${data.accuracy}</span>
            </div>
        `;
    } else if (task === 'Regression') {
        resultsContent.innerHTML = `
            <div class="result-metric">
                <span class="result-label">RMSE</span>
                <span class="result-value">${data.rmse}</span>
            </div>
            <div class="result-metric">
                <span class="result-label">R² Score</span>
                <span class="result-value">${data.r2}</span>
            </div>
        `;
    } else if (task === 'Clustering' && data.clusters) {
        let clustersHTML = '<h4 style="margin-bottom: 1rem;">Cluster Distribution</h4>';
        Object.entries(data.clusters).forEach(([cluster, count]) => {
            clustersHTML += `
                <div class="result-metric">
                    <span class="result-label">Cluster ${cluster}</span>
                    <span class="result-value">${count} samples</span>
                </div>
            `;
        });
        resultsContent.innerHTML = clustersHTML;
    } else if (task === 'Dimensionality Reduction' && data.transformed_data) {
        let html = `
            <div class="result-metric">
                <span class="result-label">Original Shape</span>
                <span class="result-value">${data.transformed_shape[0]} × ${data.transformed_shape[1]}</span>
            </div>
            <div class="result-metric">
                <span class="result-label">Showing Rows</span>
                <span class="result-value">${data.showing_rows}</span>
            </div>
            <h4 style="margin-top: 2rem; margin-bottom: 1rem; color: var(--accent-primary);">Transformed Data Preview</h4>
            <div style="overflow-x: auto; background: var(--bg-tertiary); border-radius: 12px; padding: 1rem; border: 1px solid var(--border-color);">
                <table>
                    <thead>
                        <tr>
        `;
        
        // Headers
        if (data.transformed_data.length > 0) {
            Object.keys(data.transformed_data[0]).forEach(key => {
                html += `<th>${key}</th>`;
            });
            html += `</tr></thead><tbody>`;
            
            // Rows
            data.transformed_data.forEach(row => {
                html += '<tr>';
                Object.values(row).forEach(value => {
                    html += `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`;
                });
                html += '</tr>';
            });
        }
        
        html += `</tbody></table></div>`;
        resultsContent.innerHTML = html;
    }
    
    // Display code
    if (data.code) {
        document.getElementById('codeSnippet').textContent = data.code;
    }
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    const messagesDiv = document.getElementById('chatMessages');
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.textContent = question;
    messagesDiv.appendChild(userMsg);
    
    input.value = '';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        // Add assistant message
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'chat-message assistant';
        assistantMsg.textContent = data.response;
        messagesDiv.appendChild(assistantMsg);
        
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function downloadCode() {
    const code = document.getElementById('codeSnippet').textContent;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'alrisa_code.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
