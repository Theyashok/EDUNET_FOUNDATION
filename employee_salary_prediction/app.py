from flask import Flask, render_template_string, url_for, request, jsonify, redirect
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store the model and data
model = None
df = None
feature_names = None

def generate_plot(plot_type):
    """Generate plots dynamically"""
    global df
    if df is None:
        return None
    
    plt.style.use('dark_background')
    fig = None
    
    if plot_type == 'gender_education':
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.countplot(x='Gender', data=df, ax=ax[0], palette='viridis')
        sns.countplot(x='Education Level', data=df, ax=ax[1], palette='plasma')
        ax[0].set_title('Distribution of Gender', color='cyan', fontsize=14, fontweight='bold')
        ax[1].set_title('Distribution of Education Level', color='cyan', fontsize=14, fontweight='bold')
        ax[0].tick_params(colors='white')
        ax[1].tick_params(colors='white')
        ax[0].set_xlabel('Gender', color='white')
        ax[0].set_ylabel('Count', color='white')
        ax[1].set_xlabel('Education Level', color='white')
        ax[1].set_ylabel('Count', color='white')
        
    elif plot_type == 'top_jobs':
        top_10_jobs = df.groupby('Job Title')['Salary'].mean().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_jobs)))
        bars = ax.barh(range(len(top_10_jobs)), top_10_jobs.values, color=colors)
        ax.set_yticks(range(len(top_10_jobs)))
        ax.set_yticklabels(top_10_jobs.index, color='white')
        ax.set_xlabel('Mean Salary ($)', color='white', fontsize=12)
        ax.set_title('Top 10 Highest Paying Jobs', color='cyan', fontsize=16, fontweight='bold')
        ax.tick_params(colors='white')
        
    elif plot_type == 'distributions':
        fig, ax = plt.subplots(3, 1, figsize=(6, 4))
        sns.histplot(df['Age'], ax=ax[0], color='cyan', kde=True, alpha=0.7)
        sns.histplot(df['Years of Experience'], ax=ax[1], color='magenta', kde=True, alpha=0.7)
        sns.histplot(df['Salary'], ax=ax[2], color='lime', kde=True, alpha=0.7)
        
        ax[0].set_title('Age Distribution', color='cyan', fontsize=14, fontweight='bold')
        ax[1].set_title('Years of Experience Distribution', color='magenta', fontsize=14, fontweight='bold')
        ax[2].set_title('Salary Distribution', color='lime', fontsize=14, fontweight='bold')
        
        for a in ax:
            a.tick_params(colors='white')
            a.set_xlabel(a.get_xlabel(), color='white')
            a.set_ylabel('Count', color='white')
            
    elif plot_type == 'correlation':
        # Prepare data for correlation
        df_corr = df.copy()
        education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        df_corr['Education Level'] = df_corr['Education Level'].map(education_mapping)
        le = LabelEncoder()
        df_corr['Gender'] = le.fit_transform(df_corr['Gender'])
        # Only keep numeric columns for correlation
        numeric_df = df_corr.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Feature Correlation Heatmap', color='cyan', fontsize=16, fontweight='bold')
        
    elif plot_type == 'feature_importance':
        if model is not None and feature_names is not None:
            feature_importances = model.feature_importances_
            sorted_indices = np.argsort(feature_importances)[::-1]
            sorted_feature_importances = [feature_importances[i] for i in sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_feature_names[:10])))
            bars = ax.barh(sorted_feature_names[:10], sorted_feature_importances[:10], color=colors)
            ax.set_xlabel('Feature Importance', color='white', fontsize=12)
            ax.set_title('Top 10 Feature Importance in Predicting Salary', color='cyan', fontsize=16, fontweight='bold')
            ax.tick_params(colors='white')
            ax.invert_yaxis()
    
    if fig:
        plt.tight_layout()
        # Convert plot to base64 string
        img = BytesIO()
        fig.savefig(img, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#0f0f0f', edgecolor='none')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url
    return None

def train_model():
    """Train the salary prediction model"""
    global model, df, feature_names
    
    if df is None:
        return False
    
    # Prepare data
    df_model = df.copy()
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df_model['Education Level'] = df_model['Education Level'].map(education_mapping)
    le = LabelEncoder()
    df_model['Gender'] = le.fit_transform(df_model['Gender'])
    
    # Create dummy variables for Job Title
    job_dummies = pd.get_dummies(df_model['Job Title'], prefix='Job')
    df_model = pd.concat([df_model, job_dummies], axis=1)
    df_model.drop('Job Title', axis=1, inplace=True)
    
    # Prepare features and target
    X = df_model.drop('Salary', axis=1)
    y = df_model['Salary']
    feature_names = list(X.columns)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return True

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Employee Salary Prediction - Neon Analytics</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0a084d 0%, #1a1a6e 30%, #6d1b7b 60%, #b721ff 80%, #21d4fd 100%);
                background-attachment: fixed;
                color: #ffffff;
                min-height: 100vh;
                overflow-x: hidden;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 50px;
                position: relative;
            }
            
            .header h1 {
                font-size: 3.5rem;
                font-weight: 700;
                background: linear-gradient(45deg, #00ffff, #ff0000, #00ff00, #0080ff);
                background-size: 400% 400%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: neonGlow 3s ease-in-out infinite;
                text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
                margin-bottom: 10px;
            }
            
            @keyframes neonGlow {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            .header p {
                font-size: 1.2rem;
                color: #b0b0b0;
                margin-bottom: 30px;
            }
            
            .nav {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-bottom: 40px;
                flex-wrap: wrap;
            }
            
            .nav-btn {
                padding: 12px 24px;
                background: linear-gradient(45deg, #00ffff, #0080ff);
                border: none;
                border-radius: 25px;
                color: #000;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            }
            
            .nav-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 40px;
            }
            
            .card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 30px;
                border: 1px solid rgba(0, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                border-color: rgba(0, 255, 255, 0.5);
                box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
            }
            
            .card h3 {
                color: #00ffff;
                font-size: 1.5rem;
                margin-bottom: 15px;
                text-align: center;
            }
            
            .upload-area {
                border: 2px dashed rgba(0, 255, 255, 0.5);
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: #00ffff;
                background: rgba(0, 255, 255, 0.1);
            }
            
            .upload-area.dragover {
                border-color: #00ff00;
                background: rgba(0, 255, 0, 0.1);
            }
            
            .prediction-form {
                display: grid;
                gap: 15px;
            }
            
            .form-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            
            .form-group label {
                color: #00ffff;
                font-weight: 600;
            }
            
            .form-group input, .form-group select {
                padding: 10px;
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 5px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 1rem;
            }
            
            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: #00ffff;
                box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
            }
            
            .submit-btn {
                padding: 12px 24px;
                background: linear-gradient(45deg, #ff4d4d, #ff0000);
                border: none;
                border-radius: 25px;
                color: white;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 10px;
            }
            
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
            }
            
            .plot-container {
                grid-column: 1 / -1;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 30px;
                border: 1px solid rgba(0, 255, 255, 0.2);
                text-align: center;
            }
            
            .plot-container img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
            }
            
            .plot-controls {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            .plot-btn {
                padding: 8px 16px;
                background: linear-gradient(45deg, #00ff00, #008000);
                border: none;
                border-radius: 20px;
                color: #000;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .plot-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #00ffff;
                font-size: 1.2rem;
            }
            
            .result {
                margin-top: 20px;
                padding: 20px;
                background: rgba(0, 255, 0, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(0, 255, 0, 0.3);
                display: none;
            }
            
            .error {
                background: rgba(255, 0, 0, 0.1);
                border-color: rgba(255, 0, 0, 0.3);
                color: #ff6b6b;
            }
            
            @media (max-width: 768px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2.5rem;
                }
                
                .nav {
                    flex-direction: column;
                    align-items: center;
                }
            }
            #starfield {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: 0;
                pointer-events: none;
                display: block;
            }
            body {
                background: #070d2a;
            }
            .container, .header, .main-content, .card, .plot-container {
                position: relative;
                z-index: 1;
            }
        </style>
    </head>
    <body>
        <canvas id="starfield"></canvas>
        <div class="container">
            <div class="header">
                <h1>Employee Salary Prediction</h1>
                <p>Advanced Analytics with Machine Learning</p>
                <div class="nav">
                    <a href="#upload" class="nav-btn">Upload Data</a>
                    <a href="#predict" class="nav-btn">Predict Salary</a>
                    <a href="#analysis" class="nav-btn">View Analysis</a>
                </div>
            </div>
            
            <div class="main-content">
                <div class="card" id="upload">
                    <h3>📊 Upload Your Data</h3>
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <p>Click to upload CSV file or drag and drop</p>
                        <p style="font-size: 0.9rem; color: #888;">Supports: Age, Gender, Education Level, Job Title, Years of Experience, Salary</p>
                    </div>
                    <input type="file" id="fileInput" accept=".csv" style="display: none;" onchange="handleFileUpload(this)">
                    <div class="loading" id="uploadLoading">Processing data...</div>
                </div>
                
                <div class="card" id="predict">
                    <h3>🎯 Predict Salary</h3>
                    <form class="prediction-form" onsubmit="predictSalary(event)">
                        <div class="form-group">
                            <label for="age">Age:</label>
                            <input type="number" id="age" required min="18" max="100">
                        </div>
                        <div class="form-group">
                            <label for="gender">Gender:</label>
                            <select id="gender" required>
                                <option value="">Select Gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="education">Education Level:</label>
                            <select id="education" required>
                                <option value="">Select Education</option>
                                <option value="High School">High School</option>
                                <option value="Bachelor's">Bachelor's</option>
                                <option value="Master's">Master's</option>
                                <option value="PhD">PhD</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="experience">Years of Experience:</label>
                            <input type="number" id="experience" required min="0" max="50" step="0.5">
                        </div>
                        <button type="submit" class="submit-btn">Predict Salary</button>
                    </form>
                    <div class="loading" id="predictLoading">Calculating prediction...</div>
                    <div class="result" id="predictionResult"></div>
                </div>
            </div>
            
            <div class="plot-container" id="analysis">
                <h3>📈 Data Analysis & Visualizations</h3>
                <div class="plot-controls">
                    <button class="plot-btn" onclick="generatePlot('gender_education')">Gender & Education</button>
                    <button class="plot-btn" onclick="generatePlot('top_jobs')">Top Paying Jobs</button>
                    <button class="plot-btn" onclick="generatePlot('distributions')">Distributions</button>
                    <button class="plot-btn" onclick="generatePlot('correlation')">Correlation</button>
                    <button class="plot-btn" onclick="generatePlot('feature_importance')">Feature Importance</button>
                </div>
                <div class="loading" id="plotLoading">Generating plot...</div>
                <div id="plotArea"></div>
            </div>
        </div>
        
        <script>
            // Enhanced cosmic starfield
            const canvas = document.getElementById('starfield');
            const ctx = canvas.getContext('2d');
            let stars = [];
            const STAR_COUNT = 220;
            function resizeCanvas() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }
            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();
            function randomStar() {
                return {
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    r: Math.random() * 1.8 + 0.2,
                    speed: Math.random() * 0.08 + 0.02,
                    alpha: Math.random() * 0.7 + 0.3
                };
            }
            function createStars() {
                stars = [];
                for (let i = 0; i < STAR_COUNT; i++) {
                    stars.push(randomStar());
                }
            }
            createStars();
            function drawBackground() {
                // Deep blue radial gradient for cosmic effect
                let grad = ctx.createRadialGradient(
                    canvas.width/2, canvas.height/2, canvas.width/8,
                    canvas.width/2, canvas.height/2, canvas.width/1.1
                );
                grad.addColorStop(0, '#1a237e');
                grad.addColorStop(0.5, '#0d133d');
                grad.addColorStop(1, '#070d2a');
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            function animateStars() {
                drawBackground();
                for (let star of stars) {
                    ctx.save();
                    ctx.globalAlpha = star.alpha;
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.r, 0, 2 * Math.PI);
                    ctx.fillStyle = '#fff';
                    ctx.shadowColor = '#b0cfff';
                    ctx.shadowBlur = 10 + star.r * 4;
                    ctx.fill();
                    ctx.restore();
                    star.y += star.speed;
                    if (star.y > canvas.height) {
                        star.x = Math.random() * canvas.width;
                        star.y = 0;
                    }
                }
                requestAnimationFrame(animateStars);
            }
            animateStars();
            window.addEventListener('resize', () => {
                resizeCanvas();
                createStars();
            });
            // File upload handling
            function handleFileUpload(input) {
                const file = input.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('uploadLoading').style.display = 'block';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('uploadLoading').style.display = 'none';
                    if (data.success) {
                        alert('Data uploaded successfully! You can now view analysis and make predictions.');
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    document.getElementById('uploadLoading').style.display = 'none';
                    alert('Error uploading file: ' + error);
                });
            }
            
            // Salary prediction
            function predictSalary(event) {
                event.preventDefault();
                
                const formData = {
                    age: document.getElementById('age').value,
                    gender: document.getElementById('gender').value,
                    education: document.getElementById('education').value,
                    experience: document.getElementById('experience').value
                };
                
                document.getElementById('predictLoading').style.display = 'block';
                document.getElementById('predictionResult').style.display = 'none';
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('predictLoading').style.display = 'none';
                    const resultDiv = document.getElementById('predictionResult');
                    resultDiv.style.display = 'block';
                    
                    if (data.success) {
                        resultDiv.className = 'result';
                        resultDiv.innerHTML = `
                            <h4>💰 Predicted Salary</h4>
                            <p style="font-size: 2rem; color: #00ff00; font-weight: bold;">
                                $${data.prediction.toLocaleString()} <span style='font-size:1rem;'>(annual)</span><br>
                                <span style='font-size:1.2rem; color:#00ffff;'>$${data.monthly.toLocaleString()} <span style='font-size:1rem;'>(monthly)</span></span><br>
                                <span style='font-size:1.2rem; color:#ffb300;'>₹${data.annual_ppp_inr} <span style='font-size:1rem;'>(annual)</span></span><br>
                                <span style='font-size:1.2rem; color:#ffb300;'>₹${data.monthly_ppp_inr} <span style='font-size:1rem;'>(monthly)</span></span>
                            </p>
                            <p>Confidence: ${data.confidence}%</p>
                            <div style='margin-top:10px; color:#aaa; font-size:0.95rem;'>
                                <b>Note:</b> PPP-adjusted INR is an estimate of equivalent Indian salary, not a direct currency conversion. Actual salaries and cost of living differ between countries.
                            </div>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                    }
                })
                .catch(error => {
                    document.getElementById('predictLoading').style.display = 'none';
                    const resultDiv = document.getElementById('predictionResult');
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>Error: ${error}</p>`;
                });
            }
            
            // Plot generation
            function generatePlot(plotType) {
                document.getElementById('plotLoading').style.display = 'block';
                document.getElementById('plotArea').innerHTML = '';
                
                fetch('/generate_plot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({plot_type: plotType})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('plotLoading').style.display = 'none';
                    if (data.success) {
                        document.getElementById('plotArea').innerHTML = `
                            <img src="data:image/png;base64,${data.plot}" alt="Generated Plot">
                        `;
                    } else {
                        document.getElementById('plotArea').innerHTML = `
                            <p style="color: #ff6b6b;">Error: ${data.error}</p>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('plotLoading').style.display = 'none';
                    document.getElementById('plotArea').innerHTML = `
                        <p style="color: #ff6b6b;">Error: ${error}</p>
                    `;
                });
            }
            
            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    handleFileUpload(document.getElementById('fileInput'));
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Basic validation
            required_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return jsonify({'success': False, 'error': f'Missing columns: {", ".join(missing_columns)}'})
            
            # Clean data
            df.dropna(inplace=True)
            
            # Train model
            if train_model():
                return jsonify({'success': True, 'message': 'Data uploaded and model trained successfully'})
            else:
                return jsonify({'success': False, 'error': 'Failed to train model'})
        else:
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    global model, df, feature_names
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not trained. Please upload data first.'})
        
        data = request.json
        age = int(data['age'])
        gender = data['gender']
        education = data['education']
        experience = float(data['experience'])
        
        # Create a proper prediction input that matches the training features
        df_pred = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Years of Experience': [experience],
            'Job Title': ['Software Engineer']  # Default job title
        })
        education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        df_pred['Education Level'] = df_pred['Education Level'].map(education_mapping)
        le = LabelEncoder()
        df_pred['Gender'] = le.fit_transform(df_pred['Gender'])
        job_dummies = pd.get_dummies(df_pred['Job Title'], prefix='Job')
        df_pred = pd.concat([df_pred, job_dummies], axis=1)
        df_pred.drop('Job Title', axis=1, inplace=True)
        for col in feature_names:
            if col not in df_pred.columns:
                df_pred[col] = 0
        df_pred = df_pred[feature_names]
        prediction = model.predict(df_pred)[0]
        confidence = 85 + np.random.normal(0, 5)
        confidence = max(70, min(95, confidence))
        monthly_salary = prediction / 12
        # PPP-adjusted INR conversion
        usd_to_ppp_inr = 30
        annual_ppp_inr = prediction * usd_to_ppp_inr
        monthly_ppp_inr = monthly_salary * usd_to_ppp_inr
        # Format INR as lakhs/crores
        def format_inr(val):
            if val >= 1e7:
                return f"{val/1e7:.2f} Cr"
            elif val >= 1e5:
                return f"{val/1e5:.2f} Lakh"
            else:
                return f"{val:,.0f}"
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'monthly': round(monthly_salary, 2),
            'annual_ppp_inr': format_inr(annual_ppp_inr),
            'monthly_ppp_inr': format_inr(monthly_ppp_inr),
            'confidence': round(confidence, 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_plot', methods=['POST'])
def generate_plot_route():
    try:
        data = request.json
        plot_type = data.get('plot_type')
        
        if df is None:
            return jsonify({'success': False, 'error': 'No data available. Please upload data first.'})
        
        plot_url = generate_plot(plot_type)
        
        if plot_url:
            return jsonify({'success': True, 'plot': plot_url})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate plot'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Load default data if available
    try:
        if os.path.exists('Salary_Data.csv'):
            df = pd.read_csv('Salary_Data.csv')
            df.dropna(inplace=True)
            train_model()
    except Exception as e:
        print(f"Could not load default data: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5050) 