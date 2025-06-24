import streamlit as st
import joblib
import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

vectorizer, model = load_models()

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "your_api_key_here")
@st.cache_resource(ttl=3600)
def get_news_articles(query=None, country='us', page_size=5):
    base_url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': NEWS_API_KEY,
        'country': country,
        'pageSize': page_size
    }
    if query:
        params['q'] = query
        
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    return []

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.write("Analyze news articles to detect potential misinformation")

tab1, tab2 = st.tabs(["Analyze Text", "Check Headlines"])

with tab1:
    st.subheader("Check Custom News Text")
    input_text = st.text_area("Paste news article text here:", "", height=200)
    
    if st.button("Analyze Text", key="analyze_text"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                transform_input = vectorizer.transform([input_text])
                prediction = model.predict(transform_input)
                proba = model.predict_proba(transform_input)[0]
                
                col1, col2 = st.columns(2)
                if prediction[0] == 1:
                    col1.success("✅ The News is Likely Real")
                else:
                    col1.error("❌ The News is Likely Fake")
                
                confidence = max(proba) * 100
                col2.metric("Confidence", f"{confidence:.1f}%")
                
                fig, ax = plt.subplots()
                ax.barh(['Fake', 'Real'], proba, color=['red', 'green'])
                ax.set_xlim(0, 1)
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)
                
                if hasattr(model, 'coef_'):
                    feature_names = vectorizer.get_feature_names_out()
                    coef = model.coef_[0]
                    top_features = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coef
                    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
                    
                    st.subheader("Key Indicators in This Text")
                    st.dataframe(top_features)
        else:
            st.warning("Please enter some text to analyze")

with tab2:
    st.subheader("Analyze Current News Headlines")
    col1, col2 = st.columns(2)
    country = col1.selectbox("Country", ['us', 'gb', 'in', 'ca', 'au'], index=0)
    category = col2.selectbox("Category", ['general', 'business', 'technology', 'science', 'health', 'entertainment', 'sports'], index=0)
    
    if st.button("Fetch & Analyze Headlines", key="fetch_news"):
        with st.spinner("Fetching and analyzing headlines..."):
            articles = get_news_articles(country=country, page_size=10)
            
            if articles:
                results = []
                for article in articles:
                    title = article.get('title', '')
                    content = article.get('content', '') or article.get('description', '')
                    text = f"{title}. {content}"
                    
                    transform_input = vectorizer.transform([text])
                    prediction = model.predict(transform_input)
                    proba = model.predict_proba(transform_input)[0]
                    confidence = max(proba) * 100
                    
                    results.append({
                        'Title': title,
                        'Source': article.get('source', {}).get('name', 'Unknown'),
                        'Published': article.get('publishedAt', '')[:10],
                        'Prediction': 'Real' if prediction[0] == 1 else 'Fake',
                        'Confidence': confidence,
                        'URL': article.get('url', '#')
                    })
                
                df = pd.DataFrame(results)
                st.dataframe(
                    df.style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == 'Fake' else 'background-color: #ccffcc',
                        subset=['Prediction']
                    ),
                    column_config={
                        "URL": st.column_config.LinkColumn("Link"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.subheader("Analysis Summary")
                real_count = len(df[df['Prediction'] == 'Real'])
                fake_count = len(df) - real_count
                
                fig, ax = plt.subplots()
                ax.pie([real_count, fake_count], labels=['Real', 'Fake'], colors=['green', 'red'], autopct='%1.1f%%')
                ax.set_title('Fake vs Real News Distribution')
                st.pyplot(fig)
                
            else:
                st.error("Failed to fetch news articles. Please check your API key or try again later.")

with st.sidebar:
    st.header("About")
    st.write("This app uses machine learning to detect potentially fake news articles.")
    st.write("Model: Logistic Regression")
    st.write("Vectorizer: TF-IDF")
    
    st.header("How to Use")
    st.write("1. Paste text in the 'Analyze Text' tab")
    st.write("2. Or check current headlines in the 'Check Headlines' tab")
    st.write("3. View analysis results with confidence scores")
    
    st.header("Disclaimer")
    st.warning("This tool provides predictions based on machine learning models and should not be considered definitive proof of news authenticity.")
