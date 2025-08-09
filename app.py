import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64 # To create download links for model files

# --- PyCaret Imports ---
# Using try-except blocks to handle potential import errors gracefully
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# --- Graphviz for Tree Visualization ---
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="Alrisa - AutoML App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def show_eda(df):
    """Displays the Exploratory Data Analysis section."""
    st.subheader("📊 Exploratory Data Analysis")
    st.write("Data Preview:")
    st.dataframe(df.head())
    st.write("Data Shape:")
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
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
    
    # Import plot_tree here to ensure it's available for the fallback
    from sklearn.tree import plot_tree

    concepts = {
        "Random Forest": ("A Random Forest is an ensemble of many decision trees. It makes predictions by averaging the output of individual trees, which reduces overfitting and improves accuracy.", "https://i.imgur.com/kh1wJjC.png"),
        "SVM": ("A Support Vector Machine (SVM) finds the optimal boundary (hyperplane) that best separates data points of different classes in the feature space.", "https://i.imgur.com/1S7fIgG.png"),
    }

    if model_choice.startswith("Auto"):
         st.info("The 'Auto (PyCaret)' option will automatically test multiple algorithms and select the best one for your dataset based on performance.")
    
    # Updated logic for Decision Trees with fallback
    elif model_choice in ["Decision Tree", "Decision Tree Regressor"] and model is not None:
        st.info("Displaying the structure of the trained Decision Tree.")
        
        # Try to use Graphviz first for a high-quality visual
        if GRAPHVIZ_AVAILABLE:
            try:
                dot_data = export_graphviz(model, out_file=None, 
                                           feature_names=feature_names,
                                           class_names=True if task == "Classification" else False,
                                           filled=True, rounded=True,
                                           special_characters=True,
                                           max_depth=4) # Limit depth for readability
                st.graphviz_chart(dot_data)
                st.caption("This is a high-quality visual representation of the decision rules learned by the model (via Graphviz).")
                return # Exit if successful
            except Exception:
                st.warning("Graphviz visualization failed. Showing a basic fallback plot.")

        # Fallback to matplotlib's plot_tree if Graphviz fails or is not available
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_tree(model, ax=ax, feature_names=feature_names, filled=True, rounded=True, max_depth=4, fontsize=10)
            st.pyplot(fig)
            st.caption("This is a basic visual representation of the decision rules (via scikit-learn's plot_tree).")
        except Exception as e:
            st.error(f"Could not generate any tree plot. Error: {e}")

    elif model_choice in concepts:
        desc, img_url = concepts[model_choice]
        st.markdown(desc)
        if img_url:
            st.image(img_url, caption=f"Conceptual Diagram for {model_choice}")
    else:
        st.info(f"Conceptual information for {model_choice} will be added soon.")


def generate_code_snippet(task, model_choice, target, features):
    """Generates a downloadable Python code snippet."""
    if model_choice == "Auto (PyCaret)":
        lib = "classification" if task == "Classification" else "regression"
        return f"""
# Code generated by Alrisa AutoML App
import pandas as pd
from pycaret.{lib} import setup, compare_models, save_model

# Load your data
df = pd.read_csv('your_data.csv')

# Setup PyCaret environment
print("Setting up PyCaret environment...")
s = setup(data=df, target='{target}', session_id=123, silent=True)

# Train and compare models
print("Training models...")
best_model = compare_models()

# Save the best model
save_model(best_model, 'best_model_pipeline')

print("Process complete. 'best_model_pipeline.pkl' is saved.")
"""
    # For sklearn models
    imports = "import pandas as pd\nfrom sklearn.model_selection import train_test_split"
    model_import, model_init = "", ""
    
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

    if model_choice in models_map:
        model_import, model_init = models_map[model_choice]

    feature_list = ', '.join([f'"{f}"' for f in features])

    if task in ["Classification", "Regression"]:
        return f"""
# Code generated by Alrisa AutoML App
{imports}
{model_import}

# Load data and define features/target
df = pd.read_csv('your_data.csv')
features = [{feature_list}]
target = '{target}'
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = {model_init}
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model Score: {{score}}")
"""
    else: # Unsupervised
        return f"""
# Code generated by Alrisa AutoML App
{imports}
{model_import}

# Load data and define features
df = pd.read_csv('your_data.csv')
features = [{feature_list}]
X = df[features]

# Initialize and fit the model
model = {model_init}
if hasattr(model, 'fit_predict'):
    labels = model.fit_predict(X)
    print("Cluster labels assigned.")
else:
    transformed_data = model.fit_transform(X)
    print("Data transformation complete.")
"""

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Alrisa Settings")
    st.info("Upload your data, choose a task, and let Alrisa do the rest.")
    
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    task = st.selectbox("1. Choose the ML Task", ["Classification", "Regression", "Clustering", "Dimensionality Reduction"])

    model_options = {
        "Classification": ["Auto (PyCaret)", "Decision Tree", "Random Forest", "SVM", "Logistic Regression"],
        "Regression": ["Auto (PyCaret)", "Decision Tree Regressor", "Random Forest Regressor", "Linear Regression", "SVR"],
        "Clustering": ["K-Means", "DBSCAN"],
        "Dimensionality Reduction": ["PCA", "t-SNE"],
    }
    
    # Disable PyCaret if not available
    if not PYCARET_AVAILABLE:
        model_options["Classification"].pop(0)
        model_options["Regression"].pop(0)
        st.warning("PyCaret is not installed. 'Auto' mode is disabled.")
        
    model_choice = st.selectbox("2. Select an Algorithm", model_options[task])

# --- Main Panel UI ---
st.title("🤖 Alrisa: The Automated Machine Learning Platform")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    show_eda(data)
    st.markdown("---")
    
    # Feature and Target selection
    st.header("⚙️ Model Configuration")
    
    if task in ["Classification", "Regression"]:
        target_column = st.selectbox("Select Your Target Column", data.columns)
        feature_columns = st.multiselect("Select Your Feature Columns", [col for col in data.columns if col != target_column], default=[col for col in data.columns if col != target_column and pd.api.types.is_numeric_dtype(data[col])])
    else: # Unsupervised
        target_column = None
        feature_columns = st.multiselect("Select Features for Analysis", [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])], default=[col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])

    if st.button("🚀 Run Analysis"):
        if not feature_columns:
            st.warning("Please select at least one feature column.")
        else:
            with st.spinner('The magic is happening... Please wait.'):
                try:
                    # --- MODEL TRAINING LOGIC ---
                    if model_choice == "Auto (PyCaret)":
                        setup_func = classification_setup if task == "Classification" else regression_setup
                        compare_func = classification_compare if task == "Classification" else regression_compare
                        pull_func = classification_pull if task == "Classification" else regression_pull
                        save_func = classification_save if task == "Classification" else regression_save
                        plot_func = classification_plot if task == "Classification" else regression_plot
                        
                        show_model_visuals(task, model_choice)
                        st.markdown("---")
                        st.header("📈 Results & Performance")

                        setup_func(data=data, target=target_column, session_id=123, silent=True, html=False)
                        st.info("PyCaret setup complete.")
                        best_model = compare_func()
                        st.success("Model comparison complete.")
                        
                        st.subheader("Best Model Performance")
                        st.dataframe(pull_func())
                        st.subheader("Best Performing Model Object")
                        st.write(best_model)
                        
                        # Show plots
                        st.subheader("Model Plots")
                        plot_types = ['confusion_matrix', 'auc'] if task == "Classification" else ['residuals', 'error']
                        for plot_type in plot_types:
                            try:
                                plot_func(best_model, plot=plot_type, display_format='streamlit')
                            except Exception:
                                pass # Some plots may not be applicable for all models
                        
                        # Save model and provide download link
                        save_func(best_model, 'best_model_pipeline')
                        with open('best_model_pipeline.pkl', 'rb') as f:
                            st.download_button('Download Best Model (PKL)', f, file_name='best_model_pipeline.pkl')
                    
                    else: # Scikit-learn path for specific models
                        df_processed = data[feature_columns + ([target_column] if target_column else [])].dropna()
                        X = df_processed[feature_columns]
                        y = df_processed[target_column] if target_column else None
                        
                        # Model Initialization
                        models = {
                            "Decision Tree": DecisionTreeClassifier(random_state=42), "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                            "Random Forest": RandomForestClassifier(random_state=42), "Random Forest Regressor": RandomForestRegressor(random_state=42), 
                            "SVM": SVC(random_state=42), "SVR": SVR(),
                            "Logistic Regression": LogisticRegression(random_state=42), "Linear Regression": LinearRegression()
                        }
                        model = models.get(model_choice) # Use .get for safety
                        
                        if task in ["Classification", "Regression"]:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            if model:
                                model.fit(X_train, y_train)
                            
                            show_model_visuals(task, model_choice, model, feature_columns)
                            st.markdown("---")
                            st.header("📈 Results & Performance")

                            st.subheader(f"{model_choice} Performance")
                            if task == "Classification":
                                y_pred = model.predict(X_test)
                                acc = accuracy_score(y_test, y_pred)
                                st.write(f"Accuracy: {acc:.3f}")
                                fig, ax = plt.subplots()
                                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_title("Confusion Matrix")
                                st.pyplot(fig)
                            else: # Regression
                                y_pred = model.predict(X_test)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                r2 = r2_score(y_test, y_pred)
                                st.write(f"RMSE: {rmse:.3f}")
                                st.write(f"R-squared: {r2:.3f}")

                        else: # Unsupervised
                            show_model_visuals(task, model_choice)
                            st.markdown("---")
                            st.header("📈 Results & Performance")

                            # Unsupervised model choices
                            if model_choice == "K-Means":
                                model = KMeans(n_clusters=3, random_state=42)
                                labels = model.fit_predict(X)
                            elif model_choice == "DBSCAN":
                                model = DBSCAN(eps=0.5, min_samples=5)
                                labels = model.fit_predict(X)
                            elif model_choice == "PCA":
                                model = PCA(n_components=2)
                                X_transformed = model.fit_transform(X)
                            elif model_choice == "t-SNE":
                                model = TSNE(n_components=2, random_state=42)
                                X_transformed = model.fit_transform(X)

                            st.subheader(f"{model_choice} Results")
                            if 'labels' in locals():
                                st.write("Cluster sizes:")
                                st.write(pd.Series(labels).value_counts())
                            if 'X_transformed' in locals():
                                st.write("Transformed Data Preview:")
                                st.dataframe(pd.DataFrame(X_transformed, columns=['Component 1', 'Component 2']).head())

                    # --- Code Snippet Section ---
                    st.markdown("---")
                    st.header("📝 Reproducible Code")
                    st.info("Use the code below in your own projects to replicate this model.")
                    code = generate_code_snippet(task, model_choice, target_column, feature_columns)
                    st.code(code, language="python")
                    st.download_button("Download Code Snippet", code, file_name="alrisa_snippet.py")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error("Please check your data and selections. For example, the target for classification should not be continuous, and features should be numeric for most models.")

else:
    st.info("Upload a CSV file from the sidebar to get started.")

