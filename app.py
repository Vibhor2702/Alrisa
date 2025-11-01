from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import json
from werkzeug.utils import secure_filename

# --- PyCaret Imports ---
try:
    from pycaret.classification import setup as classification_setup, compare_models as classification_compare, pull as classification_pull, save_model as classification_save
    from pycaret.regression import setup as regression_setup, compare_models as regression_compare, pull as regression_pull, save_model as regression_save
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Functions ---

def analyze_data(df):
    """Returns EDA data as JSON."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    eda_data = {
        'preview': df.head(10).to_dict('records'),
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'columns': df.columns.tolist(),
        'numeric_columns': numeric_cols
    }
    
    if numeric_cols:
        eda_data['summary'] = df[numeric_cols].describe().to_dict()
        eda_data['correlation'] = df[numeric_cols].corr().to_dict()
    
    return eda_data

def generate_code_snippet(task, model_choice, target, features):
    """Generates a downloadable Python code snippet."""
    if model_choice == "Auto (PyCaret)":
        lib = "classification" if task == "Classification" else "regression"
        return f"""
import pandas as pd
from pycaret.{lib} import setup, compare_models, save_model
df = pd.read_csv('your_data.csv')
s = setup(data=df, target='{target}', session_id=123, silent=True)
best_model = compare_models()
save_model(best_model, 'best_model_pipeline')
"""
    imports = "import pandas as pd\nfrom sklearn.model_selection import train_test_split"
    models_map = {
        "Random Forest": ("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(random_state=42)"),
        "Random Forest Regressor": ("from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor(random_state=42)"),
        "Logistic Regression": ("from sklearn.linear_model import LogisticRegression", "LogisticRegression(random_state=42)"),
        "Decision Tree": ("from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier(random_state=42)"),
        "Decision Tree Regressor": ("from sklearn.tree import DecisionTreeRegressor", "DecisionTreeRegressor(random_state=42)"),
        "SVM": ("from sklearn.svm import SVC", "SVC(random_state=42)"),
        "K-Means": ("from sklearn.cluster import KMeans", "KMeans(n_clusters=3, random_state=42)"),
        "PCA": ("from sklearn.decomposition import PCA", "PCA(n_components=2)")
    }
    model_import, model_init = models_map.get(model_choice, ("", ""))
    feature_list = ', '.join([f'"{f}"' for f in features])
    if task in ["Classification", "Regression"]:
        return f"""
{imports}\n{model_import}\ndf = pd.read_csv('your_data.csv')
features = [{feature_list}]\ntarget = '{target}'
X = df[features]\ny = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = {model_init}\nmodel.fit(X_train, y_train)
score = model.score(X_test, y_test)\nprint(f"Model Score: {{score}}")
"""
    else:
        return f"""
{imports}\n{model_import}\ndf = pd.read_csv('your_data.csv')
features = [{feature_list}]\nX = df[features]
model = {model_init}
if hasattr(model, 'fit_predict'):
    labels = model.fit_predict(X)
else:
    transformed_data = model.fit_transform(X)
"""

def get_chatbot_response(user_question):
    """A simple rule-based chatbot for user help."""
    user_question = user_question.lower()
    if "upload" in user_question: 
        return "To upload data, use the file upload area. The file must be a CSV."
    if "classification" in user_question: 
        return "Classification predicts a category (e.g., 'spam' or 'not spam')."
    if "regression" in user_question: 
        return "Regression predicts a number (e.g., house price)."
    if "clustering" in user_question: 
        return "Clustering finds natural groups in your data without labels."
    if "pycaret" in user_question: 
        return "'Auto (PyCaret)' automatically finds the best model for your data."
    return "I can help with questions about uploading data or terms like 'classification', 'regression', 'clustering', and 'PyCaret'. How can I assist?"

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html', pycaret_available=PYCARET_AVAILABLE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            eda_data = analyze_data(df)
            
            # Save the dataframe temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            df.to_csv(filepath, index=False)
            
            return jsonify({'success': True, 'filename': filename, 'eda': eda_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        filename = data.get('filename')
        task = data.get('task')
        model_choice = data.get('model')
        target_column = data.get('target')
        feature_columns = data.get('features', [])
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        results = {}
        
        if task in ["Classification", "Regression"]:
            df_proc = df[feature_columns + [target_column]].dropna()
            X = df_proc[feature_columns]
            y = df_proc[target_column]
            
            models_map = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "SVR": SVR(),
                "Linear Regression": LinearRegression()
            }
            
            model = models_map.get(model_choice)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            if task == "Classification":
                acc = model.score(X_test, y_test)
                results['accuracy'] = round(acc, 3)
                results['metric'] = 'Accuracy'
            else:
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                results['rmse'] = round(rmse, 3)
                results['r2'] = round(r2, 3)
                results['metric'] = 'RMSE & R²'
        
        else:  # Unsupervised
            # Filter only numeric columns
            df_proc = df[feature_columns].dropna()
            numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns selected. Please select numeric features for unsupervised learning.'}), 400
            
            X = df_proc[numeric_cols]
            
            models_map = {
                "K-Means": KMeans(n_clusters=3, random_state=42),
                "DBSCAN": DBSCAN(),
                "PCA": PCA(n_components=min(2, len(numeric_cols))),
                "t-SNE": TSNE(n_components=min(2, len(numeric_cols)), random_state=42)
            }
            
            model = models_map.get(model_choice)
            
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
                cluster_counts = pd.Series(labels).value_counts().to_dict()
                results['clusters'] = cluster_counts
            else:
                X_tf = model.fit_transform(X)
                results['transformed_shape'] = X_tf.shape
            
            results['numeric_features_used'] = numeric_cols
        
        # Generate code snippet
        code = generate_code_snippet(task, model_choice, target_column, feature_columns)
        results['code'] = code
        results['success'] = True
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    response = get_chatbot_response(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
