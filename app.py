import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# --- PyCaret Imports ---
try:
    from pycaret.classification import setup as classification_setup, compare_models as classification_compare, pull as classification_pull, save_model as classification_save, plot_model as classification_plot
    from pycaret.regression import setup as regression_setup, compare_models as regression_compare, pull as regression_pull, save_model as regression_save, plot_model as regression_plot
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# --- Graphviz for Tree Visualization (Optional) ---
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="Alrisa - AutoML App",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to load local CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the custom CSS
local_css("style.css")

# --- Helper Functions ---

def show_eda(df):
    """Displays the Exploratory Data Analysis section."""
    st.subheader("📊 Exploratory Data Analysis")
    st.write("Data Preview:")
    st.dataframe(df.head())
    st.write("Data Shape:", df.shape)
    st.write("Data Types:")
    st.dataframe(df.dtypes.rename("Data Type"))
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().rename("Missing Values"))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.write("Numeric Data Summary:")
        st.dataframe(df[numeric_cols].describe())
        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def show_model_visuals(task, model_choice, model=None, feature_names=None):
    """Displays conceptual or actual model structure visuals with a fallback."""
    st.subheader("📝 Model Structure & Concept")

    concepts = {
        "Random Forest": ("A Random Forest is an ensemble of many decision trees. It makes predictions by averaging the output of individual trees, which reduces overfitting and improves accuracy.", "https://i.imgur.com/kh1wJjC.png"),
        "SVM": ("A Support Vector Machine (SVM) finds the optimal boundary (hyperplane) that best separates data points of different classes in the feature space.", "https://i.imgur.com/1S7fIgG.png"),
    }

    if model_choice.startswith("Auto"):
        st.info("The 'Auto (PyCaret)' option will automatically test multiple algorithms and select the best one for your dataset.")
    
    elif model_choice in ["Decision Tree", "Decision Tree Regressor"] and model is not None:
        st.info("Displaying the structure of the trained Decision Tree.")
        if GRAPHVIZ_AVAILABLE:
            try:
                dot_data = export_graphviz(model, out_file=None, feature_names=feature_names, class_names=(task == "Classification"), filled=True, rounded=True, max_depth=4)
                st.graphviz_chart(dot_data)
                st.caption("High-quality tree visualization (via Graphviz).")
                return
            except Exception:
                st.warning("Graphviz visualization failed. Showing basic fallback plot.")
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_tree(model, ax=ax, feature_names=feature_names, filled=True, rounded=True, max_depth=4, fontsize=10)
            st.pyplot(fig)
            st.caption("Basic tree visualization (via scikit-learn).")
        except Exception as e:
            st.error(f"Could not generate tree plot: {e}")

    elif model_choice in concepts:
        desc, img_url = concepts[model_choice]
        st.markdown(desc)
        if img_url:
            st.image(img_url, caption=f"Conceptual Diagram for {model_choice}")
    else:
        st.info("Conceptual information will be added soon.")

def generate_code_snippet(task, model_choice, target, features):
    """Generates a downloadable Python code snippet."""
    # (This function remains the same as provided in the previous step)
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
    if "upload" in user_question: return "To upload data, use the 'Upload your input CSV file' button in the sidebar. The file must be a CSV."
    if "classification" in user_question: return "Classification predicts a category (e.g., 'spam' or 'not spam')."
    if "regression" in user_question: return "Regression predicts a number (e.g., house price)."
    if "clustering" in user_question: return "Clustering finds natural groups in your data without labels."
    if "pycaret" in user_question: return "'Auto (PyCaret)' automatically finds the best model for your data."
    return "I can help with questions about uploading data or terms like 'classification', 'regression', 'clustering', and 'PyCaret'. How can I assist?"

# --- UI Layout ---

with st.sidebar:
    st.title("Alrisa")
    st.info("Automated Machine Learning Platform")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    task = st.selectbox("1. Choose the ML Task", ["Classification", "Regression", "Clustering", "Dimensionality Reduction"])
    model_options = {
        "Classification": ["Auto (PyCaret)", "Decision Tree", "Random Forest", "SVM", "Logistic Regression"],
        "Regression": ["Auto (PyCaret)", "Decision Tree Regressor", "Random Forest Regressor", "Linear Regression", "SVR"],
        "Clustering": ["K-Means", "DBSCAN"],
        "Dimensionality Reduction": ["PCA", "t-SNE"],
    }
    if not PYCARET_AVAILABLE:
        model_options["Classification"].pop(0); model_options["Regression"].pop(0)
        st.warning("PyCaret not installed. 'Auto' mode disabled.")
    model_choice = st.selectbox("2. Select an Algorithm", model_options[task])

with st.expander("Need Help? Click to Chat!"):
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = get_chatbot_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.title("Automated Machine Learning Workflow")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    show_eda(data)
    st.markdown("---")
    st.header("⚙️ Model Configuration")
    
    if task in ["Classification", "Regression"]:
        target_column = st.selectbox("Select Target Column", data.columns)
        feature_columns = st.multiselect("Select Feature Columns", [c for c in data.columns if c != target_column], default=[c for c in data.columns if c != target_column and pd.api.types.is_numeric_dtype(data[c])])
    else:
        target_column = None
        feature_columns = st.multiselect("Select Features for Analysis", [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])], default=[c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])])

    if st.button("🚀 Run Analysis"):
        if not feature_columns:
            st.warning("Please select at least one feature column.")
        else:
            with st.spinner('The magic is happening...'):
                try:
                    if model_choice == "Auto (PyCaret)":
                        setup = classification_setup if task == "Classification" else regression_setup
                        compare = classification_compare if task == "Classification" else regression_compare
                        pull = classification_pull if task == "Classification" else regression_pull
                        save = classification_save if task == "Classification" else regression_save
                        plot = classification_plot if task == "Classification" else regression_plot
                        
                        show_model_visuals(task, model_choice)
                        st.markdown("---"); st.header("📈 Results")
                        setup(data=data, target=target_column, session_id=123, silent=True, html=False)
                        best = compare()
                        st.dataframe(pull())
                        st.write("Best Model:", best)
                        plot_types = ['confusion_matrix', 'auc'] if task == "Classification" else ['residuals', 'error']
                        for p_type in plot_types:
                            try: plot(best, plot=p_type, display_format='streamlit')
                            except Exception: pass
                        save(best, 'best_model_pipeline')
                        with open('best_model_pipeline.pkl', 'rb') as f:
                            st.download_button('Download Best Model', f, 'best_model.pkl')
                    
                    else: # Scikit-learn path
                        df_proc = data[feature_columns + ([target_column] if target_column else [])].dropna()
                        X = df_proc[feature_columns]
                        y = df_proc.get(target_column)
                        
                        models_map = {
                            "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(random_state=42),
                            "SVM": SVC(), "Logistic Regression": LogisticRegression(),
                            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42), "Random Forest Regressor": RandomForestRegressor(random_state=42),
                            "SVR": SVR(), "Linear Regression": LinearRegression(),
                            "K-Means": KMeans(n_clusters=3, random_state=42), "DBSCAN": DBSCAN(), "PCA": PCA(n_components=2), "t-SNE": TSNE()
                        }
                        model = models_map.get(model_choice)
                        
                        show_model_visuals(task, model_choice, model if "Tree" in model_choice else None, feature_columns)
                        st.markdown("---"); st.header("📈 Results")
                        
                        if y is not None: # Supervised
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model.fit(X_train, y_train)
                            if task == "Classification":
                                acc = model.score(X_test, y_test)
                                st.write(f"Accuracy: {acc:.3f}")
                            else:
                                preds = model.predict(X_test)
                                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")
                                st.write(f"R-squared: {r2_score(y_test, preds):.3f}")
                        else: # Unsupervised
                            if hasattr(model, 'fit_predict'):
                                labels = model.fit_predict(X)
                                st.write("Cluster sizes:", pd.Series(labels).value_counts())
                            else:
                                X_tf = model.fit_transform(X)
                                st.write("Transformed Data Preview:")
                                st.dataframe(pd.DataFrame(X_tf).head())
                    
                    st.markdown("---"); st.header("📝 Reproducible Code")
                    code = generate_code_snippet(task, model_choice, target_column, feature_columns)
                    st.code(code, language="python")
                    st.download_button("Download Code Snippet", code, "alrisa_snippet.py")

                except Exception as e:
                    st.error(f"An error occurred: {e}. Please check data and selections.")
else:
    st.info("Upload a CSV file from the sidebar to get started.")
