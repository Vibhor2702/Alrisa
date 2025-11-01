# Alrisa - Modern AutoML Platform# Alrisa - An Automated Machine Learning Platform



Alrisa is a modern, automated machine learning platform built with Flask. It provides an intuitive interface for performing classification, regression, clustering, and dimensionality reduction tasks.Hello there! I'm excited to share **Alrisa**, a personal project I built to create an intuitive, powerful, and aesthetically pleasing automated machine learning (AutoML) platform. My goal was to build a tool that could not only automate the tedious parts of the machine learning workflow but also serve as an educational resource for understanding how different models work.



## FeaturesThis application allows anyone, regardless of their technical expertise, to upload a dataset, run complex machine learning analyses, and receive clear, actionable insights in just a few clicks.



- 🎯 **Multi-Task Support**: Classification, Regression, Clustering, and Dimensionality Reduction## ✨ Key Features

- 🤖 **Multiple Algorithms**: Decision Trees, Random Forests, SVM, Logistic Regression, K-Means, PCA, and more

- 📊 **Automated EDA**: Instant exploratory data analysis with statistics and visualizationsI've packed Alrisa with a suite of features to make the machine learning process as seamless and insightful as possible:

- 💬 **AI Assistant**: Built-in help chatbot for ML guidance

- 🎨 **Modern UI**: Clean, minimalist design inspired by premium tech aesthetics*   **Multi-Task Support**: The app isn't limited to one type of problem. I've built in support for:

- 📝 **Code Generation**: Download reproducible Python code for your models    *   Classification: To predict categories.

    *   Regression: To predict numerical values.

## Quick Start    *   Clustering: To discover natural groupings in data.

    *   Dimensionality Reduction: To simplify complex datasets.

### Local Development*   **Automated & Manual Model Selection**:

    *   **Auto (PyCaret)**: For users who want the best model without the guesswork, this option automatically trains and compares dozens of models to find the top performer.

1. **Install Dependencies**    *   **Manual Selection**: For more hands-on analysis, users can select specific algorithms like Decision Trees, Random Forests, SVMs, and more.

```bash*   **Rich, Interactive Visualizations**: I believe that data should be seen, not just read. Alrisa provides:

pip install -r requirements.txt    *   **Automated EDA**: Instant exploratory data analysis with data summaries, correlation heatmaps, and distribution plots.

```    *   **Model Structure Views**: For models like Decision Trees, the app generates interactive tree/graph diagrams to show how the model makes decisions.

    *   **Performance Plots**: Clear confusion matrices, ROC curves, and residual plots to evaluate model performance.

2. **Run the Application***   **Downloadable Code Snippets**: To bridge the gap between automation and custom coding, every model trained generates a clean, reproducible Python code snippet that can be downloaded and used in other projects.

```bash*   **"Nothing" Inspired UI**: I took inspiration from the minimalist, tech-forward aesthetic of the "Nothing" brand, using a monochrome color scheme, a unique dot-matrix font, and clean layouts to create a visually striking user experience.

python app.py*   **Integrated Help Chatbot**: To make the app as user-friendly as possible, I built in a simple, rule-based chatbot to answer common questions about machine learning terms and app functionality.

```

## 🛠️ Tech Stack

3. **Open Browser**

Navigate to `http://localhost:8080`I built this project using a powerful stack of Python libraries and tools:



## Deployment Options*   **Core Framework**: Streamlit

*   **AutoML Engine**: PyCaret

### Option 1: Railway (Recommended)*   **Machine Learning**: Scikit-learn

*   **Data Manipulation**: Pandas, NumPy

Railway provides the easiest deployment for Python Flask apps:*   **Data Visualization**: Matplotlib, Seaborn, Graphviz

*   **Styling**: Custom HTML/CSS

1. Go to [railway.app](https://railway.app)

2. Click "New Project" → "Deploy from GitHub"## 🚀 How to Run This App Locally

3. Select your repository

4. Railway will auto-detect Flask and deploy automaticallyIf you'd like to run Alrisa on your own machine, you can get it set up with just a few steps.

5. Your app will be live with a public URL!

1.  **Clone the Repository**

### Option 2: Render    ```bash

    git clone https://github.com/your-username/Alrisa-ML-App.git

1. Go to [render.com](https://render.com)    cd Alrisa-ML-App

2. Click "New" → "Web Service"    ```

3. Connect your repository

4. Configure:2.  **Install Dependencies**

   - **Build Command**: `pip install -r requirements.txt`    I've listed all the required Python packages in the `requirements.txt` file. You can install them all with one command:

   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`    ```bash

5. Click "Create Web Service"    pip install -r requirements.txt

    ```

### Option 3: Cloudflare Pages (Static Frontend + Separate Backend)

3.  **Install Graphviz (Optional, for Tree Diagrams)**

Since Cloudflare Pages is optimized for static sites:    For the high-quality decision tree visualizations, you'll need to install the Graphviz software. You can find instructions for your OS on the [official Graphviz download page](https://graphviz.org/download/).



1. **Deploy Backend** to Railway/Render using instructions above4.  **Run the App**

2. **Update Frontend**: In `static/script.js`, update all fetch URLs to your backend URL    Once the dependencies are installed, run the following command in your terminal:

3. **Deploy to Cloudflare Pages**:    ```bash

   - Go to Cloudflare Dashboard    streamlit run app.py

   - Click "Pages" → "Create a project"    ```

   - Connect your GitHub repository    Your browser should automatically open a new tab with the running application!

   - Build settings:

     - Build command: (leave empty)## 📂 Project Structure

     - Build output directory: `/`

     - Root directory: (leave empty)Here’s how I've organized the project files:

/

## Project Structure├── app.py # The main Streamlit application script

├── requirements.txt # Python dependencies

```└── style.css # Custom CSS for the "Nothing" theme

.

├── app.py                 # Flask backend

├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # Modern UI styles
│   └── script.js         # Frontend logic
└── uploads/              # Temporary file storage
```

## Technologies Used

- **Backend**: Flask, scikit-learn, pandas, numpy
- **ML Libraries**: scikit-learn, pycaret
- **Frontend**: Vanilla JavaScript, Modern CSS
- **Design**: Minimalist dark theme with red accents

## Usage

1. **Upload Data**: Click the upload area and select a CSV file
2. **Choose Task**: Select Classification, Regression, Clustering, or Dimensionality Reduction
3. **Select Algorithm**: Pick from available ML algorithms
4. **Configure**: Choose target column and features
5. **Run Analysis**: Click "Run Analysis" to train the model
6. **View Results**: See performance metrics and generated code
7. **Download Code**: Get reproducible Python code for your model

## What's New (v2.0)

### ✅ Removed Streamlit Dependencies
- Converted from Streamlit to Flask REST API
- Now deployable on any platform

### ✅ Modern UI
- Professional, minimalist design
- Dark theme with premium red accents
- Fully responsive layout
- Smooth animations and transitions

### ✅ Fixed Issues
- Added pycaret to requirements.txt
- Removed unnecessary dependencies
- Improved error handling
- Better mobile support

## Environment Variables

No environment variables required for local development. For production:

- `PORT`: Port number (automatically set by most hosting platforms)

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT License

---

Built with ❤️ for the ML community
