import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import matplotlib

# Configuration
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit Cloud

# Load models with caching
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

vectorizer, model = load_models()

# News API functions
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_news_articles(query=None, country='us', page_size=5):
    base_url = 'https://newsapi.org/v2/top-headlines'
    headers = {'X-Api-Key': st.secrets["NEWS_API_KEY"]}
    params = {
        'country': country,
        'pageSize': page_size,
        'language': 'en'
    }
    
    if query:
        params['q'] = query
        
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            error_msg = data.get('message', 'Unknown error')
            st.error(f"News API Error: {error_msg}")
            return []
            
        return data.get('articles', [])
        
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return []

# Fallback sample data
def get_sample_articles():
    return [{
        'title': 'Sample News: Climate Accord Signed',
        'content': 'World leaders have signed a new climate agreement to reduce emissions by 2030.',
        'source': {'name': 'Sample News'},
        'publishedAt': '2023-01-01T00:00:00Z',
        'url': 'https://example.com'
    }]

# App UI
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.write("Analyze news articles for potential misinformation")

tab1, tab2 = st.tabs(["Analyze Text", "Check Headlines"])

with tab1:
    st.subheader("Check Custom News Text")
    input_text = st.text_area("Paste news article text here:", "", height=200)
    
    if st.button("Analyze Text"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    features = vectorizer.transform([input_text])
                    prediction = model.predict(features)
                    proba = model.predict_proba(features)[0]
                    
                    col1, col2 = st.columns(2)
                    if prediction[0] == 1:
                        col1.success("✅ Likely Real News")
                    else:
                        col1.error("❌ Likely Fake News")
                    
                    confidence = max(proba) * 100
                    col2.metric("Confidence", f"{confidence:.1f}%")
                    
                    fig, ax = plt.subplots()
                    ax.barh(['Fake', 'Real'], proba, color=['red', 'green'])
                    ax.set_title('Prediction Probabilities')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter text to analyze")

with tab2:
    st.subheader("Analyze Current Headlines")
    col1, col2 = st.columns(2)
    country = col1.selectbox("Country", ['us', 'gb', 'ca', 'au'], index=0)
    category = col2.selectbox("Category", ['general', 'business', 'technology', 'health'], index=0)
    
    if st.button("Fetch Headlines"):
        if 'NEWS_API_KEY' not in st.secrets:
            st.error("API key not configured")
            st.stop()
            
        with st.spinner("Fetching latest headlines..."):
            articles = get_news_articles(country=country) or get_sample_articles()
            
            if articles:
                results = []
                for article in articles:
                    text = f"{article.get('title', '')}. {article.get('content', '') or article.get('description', '')}"
                    
                    try:
                        features = vectorizer.transform([text])
                        prediction = model.predict(features)
                        proba = model.predict_proba(features)[0]
                        
                        results.append({
                            'Title': article.get('title', 'No title'),
                            'Source': article.get('source', {}).get('name', 'Unknown'),
                            'Prediction': 'Real' if prediction[0] == 1 else 'Fake',
                            'Confidence': max(proba) * 100,
                            'URL': article.get('url', '#')
                        })
                    except Exception as e:
                        continue
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(
                        df.style.applymap(
                            lambda x: 'background-color: #ffcccc' if x == 'Fake' else 'background-color: #ccffcc',
                            subset=['Prediction']
                        ),
                        column_config={
                            "URL": st.column_config.LinkColumn("Link"),
                            "Confidence": st.column_config.ProgressColumn(
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.warning("No valid articles to analyze")
            else:
                st.error("Failed to fetch articles")

with st.sidebar:
    st.header("API Status")
    if 'NEWS_API_KEY' in st.secrets:
        st.success("✅ API Key Configured")
        st.caption(f"Key: `{st.secrets['NEWS_API_KEY'][:4]}...{st.secrets['NEWS_API_KEY'][-4:]}`")
    else:
        st.error("❌ Missing API Key")
        
    st.header("How to Use")
    st.write("1. Paste text or fetch headlines")
    st.write("2. View prediction and confidence")
    st.write("3. Click article links to verify")
    
    st.header("Disclaimer")
    st.warning("Predictions are algorithmic estimates, not definitive judgments")
