import streamlit as st
import pandas as pd
from pycaret.classification import setup as classification_setup, compare_models as classification_compare, pull as classification_pull, save_model as classification_save
from pycaret.regression import setup as regression_setup, compare_models as regression_compare, pull as regression_pull, save_model as regression_save
import os

# Set page configuration
st.set_page_config(
    page_title="AutoML Web App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for file upload and settings
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML Settings")
    st.info("Upload your data and choose your settings to get started.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    # Machine learning type selection
    ml_type = st.selectbox("Choose the ML Problem Type", ["Classification", "Regression"])
    
    # Start button
    start_button = st.button("Run Analysis")

# Main panel
st.title("🤖 Automated Machine Learning Platform")

if start_button:
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Dataset")
            st.dataframe(data.head())

            # Target column selection
            target_column = st.selectbox("Select the Target Column", data.columns)

            if st.button("Start Modeling"):
                with st.spinner('Training models... This may take a moment.'):
                    if ml_type == "Classification":
                        # Setup for classification
                        classification_setup(data=data, target=target_column, session_id=123)
                        st.info("Classification setup complete.")
                        
                        # Compare models
                        best_model = classification_compare()
                        st.success("Model comparison complete.")
                        
                        # Display results
                        st.subheader("Best Model Performance")
                        setup_df = classification_pull()
                        st.dataframe(setup_df)
                        
                        st.subheader("Best Performing Model")
                        st.write(best_model)
                        
                        # Save the model
                        classification_save(best_model, 'best_model')

                    elif ml_type == "Regression":
                        # Setup for regression
                        regression_setup(data=data, target=target_column, session_id=123)
                        st.info("Regression setup complete.")
                        
                        # Compare models
                        best_model = regression_compare()
                        st.success("Model comparison complete.")
                        
                        # Display results
                        st.subheader("Best Model Performance")
                        setup_df = regression_pull()
                        st.dataframe(setup_df)
                        
                        st.subheader("Best Performing Model")
                        st.write(best_model)
                        
                        # Save the model
                        regression_save(best_model, 'best_model')

                # Provide download link for the saved model
                if os.path.exists('best_model.pkl'):
                    with open('best_model.pkl', 'rb') as f:
                        st.download_button('Download Best Model', f, file_name='best_model.pkl')
                        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a CSV file to begin.")

else:
    st.info("Upload a file and click 'Run Analysis' to start.")

