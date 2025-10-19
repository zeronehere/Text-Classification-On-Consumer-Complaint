import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon="🏦",
    layout="wide"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

# Text Preprocessor Class (recreated)
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple tokenization (avoid word_tokenize issues)
        tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

# Load models
@st.cache_resource
def load_models():
    models_dir = Path("models")
    model = joblib.load(models_dir / "best_model.pkl")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    target_names = joblib.load(models_dir / "target_names.pkl")
    preprocessor = TextPreprocessor()  # Create new instance instead of loading
    
    return model, vectorizer, preprocessor, target_names

# Prediction function
def predict_complaint(text, model, vectorizer, preprocessor, target_names):
    cleaned_text = preprocessor.clean_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]
    
    return {
        'prediction': prediction,
        'category': target_names[prediction],
        'confidence': confidence,
        'probabilities': probabilities,
        'cleaned_text': cleaned_text
    }

# Main app
def main():
    st.title("🏦 Consumer Complaint Classification System")
    st.markdown("### Classify complaints into 4 financial categories with **97.29% accuracy**")
    
    # Load models
    try:
        model, vectorizer, preprocessor, target_names = load_models()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Accuracy", "97.29%")
        st.metric("Training Samples", "904,255")
        st.metric("Model", type(model).__name__)
        
        st.header("🎯 Categories")
        for target_id, name in target_names.items():
            st.write(f"**{target_id}**: {name}")
    
    # Main content
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction"])
    
    with tab1:
        st.header("Single Complaint Classification")
        
        # Text input
        complaint_text = st.text_area(
            "Enter consumer complaint text:",
            height=150,
            placeholder="Example: My credit report has incorrect information that needs to be fixed..."
        )
        
        if st.button("🚀 Classify Complaint", type="primary"):
            if complaint_text.strip():
                with st.spinner("Analyzing complaint..."):
                    result = predict_complaint(complaint_text, model, vectorizer, preprocessor, target_names)
                
                # Display results
                st.success("✅ Classification Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Predicted Category",
                        value=result['category'],
                        delta=f"{result['confidence']:.2%} confidence"
                    )
                
                with col2:
                    st.metric(
                        label="Confidence Score",
                        value=f"{result['confidence']:.2%}"
                    )
                
                # Probabilities chart
                st.subheader("📈 Prediction Probabilities")
                prob_data = pd.DataFrame({
                    'Category': [target_names[i] for i in range(len(result['probabilities']))],
                    'Probability': result['probabilities']
                }).sort_values('Probability', ascending=False)
                
                st.bar_chart(prob_data.set_index('Category'))
                
            else:
                st.warning("⚠️ Please enter some complaint text.")
    
    with tab2:
        st.header("Batch Complaint Classification")
        
        # Sample complaints
        sample_complaints = """Credit report shows accounts that don't belong to me
Debt collector called me multiple times today
Personal loan interest rate increased suddenly
Mortgage application was denied without explanation"""
        
        batch_text = st.text_area(
            "Enter multiple complaints (one per line):",
            height=200,
            value=sample_complaints
        )
        
        if st.button("📦 Process Batch", type="primary"):
            if batch_text.strip():
                complaints = [line.strip() for line in batch_text.split('\n') if line.strip()]
                results = []
                
                for complaint in complaints:
                    result = predict_complaint(complaint, model, vectorizer, preprocessor, target_names)
                    results.append({
                        'Complaint': complaint[:80] + "..." if len(complaint) > 80 else complaint,
                        'Category': result['category'],
                        'Confidence': f"{result['confidence']:.2%}"
                    })
                
                st.success(f"✅ Processed {len(results)} complaints!")
                st.dataframe(pd.DataFrame(results))

if __name__ == "__main__":
    main()