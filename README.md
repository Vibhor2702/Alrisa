# Alrisa - An Automated Machine Learning Platform

Hello there! I'm excited to share **Alrisa**, a personal project I built to create an intuitive, powerful, and aesthetically pleasing automated machine learning (AutoML) platform. My goal was to build a tool that could not only automate the tedious parts of the machine learning workflow but also serve as an educational resource for understanding how different models work.

This application allows anyone, regardless of their technical expertise, to upload a dataset, run complex machine learning analyses, and receive clear, actionable insights in just a few clicks.

## ✨ Key Features

I've packed Alrisa with a suite of features to make the machine learning process as seamless and insightful as possible:

*   **Multi-Task Support**: The app isn't limited to one type of problem. I've built in support for:
    *   Classification: To predict categories.
    *   Regression: To predict numerical values.
    *   Clustering: To discover natural groupings in data.
    *   Dimensionality Reduction: To simplify complex datasets.
*   **Automated & Manual Model Selection**:
    *   **Auto (PyCaret)**: For users who want the best model without the guesswork, this option automatically trains and compares dozens of models to find the top performer.
    *   **Manual Selection**: For more hands-on analysis, users can select specific algorithms like Decision Trees, Random Forests, SVMs, and more.
*   **Rich, Interactive Visualizations**: I believe that data should be seen, not just read. Alrisa provides:
    *   **Automated EDA**: Instant exploratory data analysis with data summaries, correlation heatmaps, and distribution plots.
    *   **Model Structure Views**: For models like Decision Trees, the app generates interactive tree/graph diagrams to show how the model makes decisions.
    *   **Performance Plots**: Clear confusion matrices, ROC curves, and residual plots to evaluate model performance.
*   **Downloadable Code Snippets**: To bridge the gap between automation and custom coding, every model trained generates a clean, reproducible Python code snippet that can be downloaded and used in other projects.
*   **"Nothing" Inspired UI**: I took inspiration from the minimalist, tech-forward aesthetic of the "Nothing" brand, using a monochrome color scheme, a unique dot-matrix font, and clean layouts to create a visually striking user experience.
*   **Integrated Help Chatbot**: To make the app as user-friendly as possible, I built in a simple, rule-based chatbot to answer common questions about machine learning terms and app functionality.

## 🛠️ Tech Stack

I built this project using a powerful stack of Python libraries and tools:

*   **Core Framework**: Streamlit
*   **AutoML Engine**: PyCaret
*   **Machine Learning**: Scikit-learn
*   **Data Manipulation**: Pandas, NumPy
*   **Data Visualization**: Matplotlib, Seaborn, Graphviz
*   **Styling**: Custom HTML/CSS

## 🚀 How to Run This App Locally

If you'd like to run Alrisa on your own machine, you can get it set up with just a few steps.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/Alrisa-ML-App.git
    cd Alrisa-ML-App
    ```

2.  **Install Dependencies**
    I've listed all the required Python packages in the `requirements.txt` file. You can install them all with one command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Graphviz (Optional, for Tree Diagrams)**
    For the high-quality decision tree visualizations, you'll need to install the Graphviz software. You can find instructions for your OS on the [official Graphviz download page](https://graphviz.org/download/).

4.  **Run the App**
    Once the dependencies are installed, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open a new tab with the running application!

## 📂 Project Structure

Here’s how I've organized the project files:
/
├── app.py # The main Streamlit application script
├── requirements.txt # Python dependencies
└── style.css # Custom CSS for the "Nothing" theme


