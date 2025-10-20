# Consumer Complaint Text Classification System

📊 **Project Overview**
A machine learning system that automatically classifies consumer financial complaints into 4 categories using real-world data from the Consumer Financial Protection Bureau (CFPB).

---

### 🎯 Classification Categories
0: Credit reporting, repair, or other  
1: Debt collection  
2: Consumer Loan  
3: Mortgage  

---

### 📈 Model Performance
- **Best Model:** Random Forest (achieved highest accuracy)  
- **Training Samples:** 904,255 complaints  
- **Test Samples:** 226,064 complaints  
- **Features:** 1,028 TF-IDF features  

---

### 🔗 Dataset Source
- **Official Source:** Consumer Complaint Database  
- **Provider:** Consumer Financial Protection Bureau (CFPB)  
- **Data Type:** Consumer financial complaints  
- **Size:** 6GB+ (raw dataset)

---

### 🛠️ Technical Stack
- **Programming Language:** Python 3.8+  
- **Machine Learning:** Scikit-learn  
- **Text Processing:** NLTK, TF-IDF Vectorization  
- **Visualization:** Matplotlib, Seaborn  
- **Web Interface:** Streamlit  
- **Environment Management:** Virtual Environment  

---

### 📁 Project Structure
```
Text-Classification-On-Consumer-Complaint/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── target_names.pkl
│   └── pipeline_info.json
├── notebooks/
│   ├── model_training.ipynb
│   └── text_preprocessing.ipynb
├── src/
│   └── data_loader.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

### ⚠️ Files Not Pushed to Git
- `data/raw/consumer_complaints.csv` (6GB original dataset)  
- `data/processed/complaints_cleaned_ready.csv`  
- `models/` directory  
- `text_classification_env/` (virtual environment)

---

### 🚀 Installation & Setup
1. **Clone the Repository**
```bash
git clone <repository-url>
cd Text-Classification-On-Consumer-Complaint
```
2. **Create Virtual Environment**
```bash
python -m venv text_classification_env
```
3. **Activate Environment**
- Windows:
```bash
text_classification_env\Scripts\activate
```
- Mac/Linux:
```bash
source text_classification_env/bin/activate
```
4. **Install Dependencies**
```bash
pip install -r requirements.txt
```
5. **Download NLTK Data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
6. **Download Dataset**
- Visit CFPB Consumer Complaint Database  
- Download full dataset (CSV)  
- Place in: `data/raw/consumer_complaints.csv`

---

### 🔧 Usage Instructions
**Step 1: Data Loading & EDA**
```bash
jupyter notebook src/data_loader.ipynb
```
**Step 2: Text Preprocessing**
```bash
jupyter notebook notebooks/text_preprocessing.ipynb
```
**Step 3: Model Training & Demo**
```bash
jupyter notebook notebooks/model_training.ipynb
```
**Step 4: Web Application**
```bash
streamlit run app.py
```
Access at: [http://localhost:8501](http://localhost:8501)

---

### 🧠 Machine Learning Approach
**Models Evaluated:**
- Logistic Regression  
- Random Forest  
- Multinomial Naive Bayes  

**Pipeline Steps:**
- Text cleaning, lemmatization  
- TF-IDF feature extraction (unigrams + bigrams)  
- Model training (80-20 split)  
- Evaluation and comparison  

---

### 🌐 Web Application Features
- Real-time single complaint classification  
- Batch processing for multiple complaints  
- Probability charts and metrics visualization  
- Export predictions as CSV  

---

### 🔮 Demo Predictions
- Credit Reporting: "My credit report shows fraudulent accounts"  
- Debt Collection: "Debt collector calls me daily"  
- Consumer Loan: "Loan interest increased from 8% to 15%"  
- Mortgage: "Mortgage application denied despite good credit"  

---

### 🎯 Key Achievements
- Random Forest achieved best accuracy  
- Handles over 1M+ complaints  
- Interactive Streamlit web interface  
- Scalable and production-ready  

---

### 📝 Notebooks Overview
- `src/data_loader.ipynb`: Data loading and EDA  
- `notebooks/text_preprocessing.ipynb`: Text cleaning and feature extraction  
- `notebooks/model_training.ipynb`: Model training and evaluation  

---

### 🆘 Troubleshooting
- Dataset too large → Use sampling method  
- Memory errors → Reduce sample size  
- Model load errors → Re-run model training notebook  

---


---

