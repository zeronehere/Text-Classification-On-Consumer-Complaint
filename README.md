Consumer Complaint Text Classification System
📊 Project Overview
A machine learning system that automatically classifies consumer financial complaints into 4 categories using real-world data from the Consumer Financial Protection Bureau (CFPB).

🎯 Classification Categories
0: Credit reporting, repair, or other

1: Debt collection

2: Consumer Loan

3: Mortgage

📈 Model Performance
Best Model: Random Forest (achieved highest accuracy)

Training Samples: 904,255 complaints

Test Samples: 226,064 complaints

Features: 1,028 TF-IDF features

🔗 Dataset Source
Official Source: Consumer Complaint Database
Provider: Consumer Financial Protection Bureau (CFPB)
Data Type: Consumer financial complaints
Size: 6GB+ (raw dataset)

🛠️ Technical Stack
Programming Language: Python 3.8+

Machine Learning: Scikit-learn

Text Processing: NLTK, TF-IDF Vectorization

Visualization: Matplotlib, Seaborn

Web Interface: Streamlit

Environment Management: Virtual Environment

📁 Project Structure
text
Text-Classification-On-Consumer-Complaint/
├── 📁 data/
│   ├── 📁 raw/                 # Original dataset (not pushed to Git)
│   └── 📁 processed/           # Cleaned and processed data
├── 📁 models/                  # Trained models and pipeline (not pushed to Git)
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── target_names.pkl
│   └── pipeline_info.json
├── 📁 notebooks/
│   ├── model_training.ipynb    # Model training, evaluation & demo predictions
│   └── text_preprocessing.ipynb # Text cleaning and feature engineering
├── 📁 src/
│   └── data_loader.ipynb       # Data loading and EDA
├── 📄 app.py                   # Streamlit web application
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore              # Git ignore rules
└── 📄 README.md               # Project documentation
⚠️ Files Not Pushed to Git
The following files are excluded from the repository due to size constraints:

data/raw/consumer_complaints.csv (6GB original dataset - download from CFPB)

data/processed/complaints_cleaned_ready.csv (processed dataset)

models/ directory (trained model binaries)

text_classification_env/ (Python virtual environment)

🚀 Installation & Setup
1. Clone the Repository
bash
git clone <repository-url>
cd Text-Classification-On-Consumer-Complaint
2. Create Virtual Environment
bash
python -m venv text_classification_env
3. Activate Environment
Windows:

bash
text_classification_env\Scripts\activate
Mac/Linux:

bash
source text_classification_env/bin/activate
4. Install Dependencies
bash
pip install -r requirements.txt
5. Download NLTK Data
bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
6. Download Dataset
Visit CFPB Consumer Complaint Database

Download the full dataset (CSV format)

Place the file in data/raw/consumer_complaints.csv

🔧 Usage Instructions
Step 1: Data Loading & EDA
Run the data loader notebook first:

bash
jupyter notebook src/data_loader.ipynb
Downloads and samples data from CFPB

Performs exploratory data analysis

Creates balanced dataset for training

Step 2: Text Preprocessing
bash
jupyter notebook notebooks/text_preprocessing.ipynb
Text cleaning and normalization

TF-IDF feature engineering

Train-test split preparation

Step 3: Model Training & Demo
bash
jupyter notebook notebooks/model_training.ipynb
Trains 3 classification models

Compares performance and selects best model

Includes demo predictions on sample texts

Saves trained models

Step 4: Web Application
bash
streamlit run app.py
Access the web interface at http://localhost:8501

🧠 Machine Learning Approach
Models Evaluated
Logistic Regression - Fast and interpretable

Random Forest - Best performing model

Multinomial Naive Bayes - Efficient for text data

Technical Pipeline
Text Cleaning: Lowercasing, special character removal, lemmatization

Feature Engineering: TF-IDF with 5,000 features, unigrams and bigrams

Model Training: 80-20 stratified split, class weight balancing

Evaluation: Comprehensive model comparison

🌐 Web Application Features
The Streamlit app (app.py) provides:

Single Complaint Classification: Real-time prediction with confidence scores

Batch Processing: Classify multiple complaints at once

Visualizations: Probability charts and performance metrics

Export Results: Download predictions as CSV

🔮 Demo Predictions
Example predictions from the model:

Credit Reporting: "My credit report shows fraudulent accounts and wrong personal information"
Debt Collection: "Debt collector calls me 20 times daily with threats and harassment"
Consumer Loan: "Personal loan interest rate increased from 8% to 15% without notice"
Mortgage: "Mortgage application denied due to credit score despite good income"

🎯 Key Achievements
High Performance: Random Forest achieved best accuracy

Scalable: Efficiently processes 1M+ complaints

Production Ready: Complete deployment pipeline

User-Friendly: Interactive web interface

Real-time: Fast predictions with confidence scores

📝 Notebooks Overview
src/data_loader.ipynb
Downloads data from CFPB API

Handles 6GB+ dataset with smart chunking

Creates balanced training samples

Exploratory data analysis and visualizations

notebooks/text_preprocessing.ipynb
Text cleaning and normalization

TF-IDF feature engineering

Data preparation for model training

notebooks/model_training.ipynb
Trains and compares 3 classification models

Model evaluation and performance analysis

Includes demo predictions section

Saves trained models for deployment

⚡ Quick Start
For immediate testing without downloading full dataset:

Run the notebooks in order

Models will work with available sample data

Use the Streamlit app for interactive predictions

🆘 Troubleshooting
Common Issues:

Dataset too large: Use the provided sampling method

Memory errors: Reduce sample size in data loader

Model loading issues: Re-run model training notebook

📄 License
Educational project for demonstration purposes. Dataset provided by CFPB under open data policies.

👥 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Create a Pull Request
