import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Play Store Reviews Analyzer",
    page_icon="📱",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google_play_scraper import Sort, reviews
import json
import os
import re
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import plotly.graph_objects as go
import ssl
import certifi
from reviewer.summary import build_executive_summary, format_headline

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')  # For POS tagging

download_nltk_resources()

# Common feature request phrases
FEATURE_REQUEST_PHRASES = [
    'add', 'please add', 'would be nice', 'should add', 'need',
    'missing', 'feature request', 'feature', 'implement', 'include',
    'should have', 'would like', 'want', 'needs', 'could use', 
    'would love', 'wish', 'hope', 'update'
]

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
NEGATION_WORDS = {'not', 'no', 'nor'}

st.title("📱 Play Store Reviews Analyzer")

# App configuration
with st.sidebar:
    st.header("App Configuration")
    app_id = st.text_input("Play Store App ID", placeholder="com.example.app").strip()
    
    count = st.number_input("Number of reviews to fetch", min_value=10, max_value=10000, value=100)
    
    lang = st.selectbox("Language", ["en", "es", "fr", "de", "it"], index=0)
    
    country = st.selectbox("Country", ["us", "es", "fr", "de", "it"], index=0)
    
    sort_order = st.selectbox(
        "Sort Order", 
        ["Most Relevant", "Newest"],
        index=0
    )
    
    sort_map = {
        "Most Relevant": Sort.MOST_RELEVANT,
        "Newest": Sort.NEWEST
    }
    
    fetch_button = st.button("Fetch Reviews")

# Function to scrape reviews
def scrape_reviews(app_id, count, lang, country, sort_order):
    try:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort_map[sort_order],
            count=count
        )
        return result
    except Exception as e:
        st.error(f"Error fetching reviews: {str(e)}")
        return None

# Function to save reviews locally
def save_reviews(reviews_data, app_id):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{app_id}_{timestamp}.json"
    
    # Custom JSON encoder to handle datetime objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    with open(filename, 'w') as f:
        json.dump(reviews_data, f, indent=4, cls=DateTimeEncoder)
    
    csv_filename = f"data/{app_id}_{timestamp}.csv"
    df = pd.DataFrame(reviews_data)
    df.to_csv(csv_filename, index=False)
    
    return filename, csv_filename

# Function to preprocess text
# keep_negations=True keeps words like "not" so "not working" survives (used for themes)
def preprocess_text(text, keep_negations=False):
    if not isinstance(text, str):
        return ""

    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    removed = stop_words - NEGATION_WORDS if keep_negations else stop_words
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in removed]
    
    return " ".join(cleaned_tokens)

# Function to perform sentiment analysis
def analyze_sentiment(reviews_df):
    # Apply sentiment analysis
    reviews_df['sentiment_score'] = reviews_df['content'].apply(
        lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    
    # Categorize sentiments
    reviews_df['sentiment'] = pd.cut(
        reviews_df['sentiment_score'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Add preprocessed text for word frequency analysis
    reviews_df['processed_text'] = reviews_df['content'].apply(preprocess_text)
    reviews_df['theme_text'] = reviews_df['content'].apply(
        lambda x: preprocess_text(x, keep_negations=True)
    )

    return reviews_df

# Function to extract top words and phrases
def extract_key_phrases(df, sentiment_filter=None):
    if sentiment_filter:
        filtered_df = df[df['sentiment'] == sentiment_filter]
    else:
        filtered_df = df
    
    # Join all processed text
    all_text = ' '.join(filtered_df['processed_text'].dropna())
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    
    # Extract top words using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=20)
    X = vectorizer.fit_transform(filtered_df['processed_text'].dropna())
    words = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    
    top_words_df = pd.DataFrame({'word': words, 'count': counts})
    top_words_df = top_words_df.sort_values('count', ascending=False)
    
    return wordcloud, top_words_df

# Function to extract feature requests
def detect_feature_requests(reviews_df):
    # Create a new column for feature request detection
    reviews_df['is_feature_request'] = False
    
    # Regex pattern for feature requests
    feature_pattern = r'\b(' + '|'.join(FEATURE_REQUEST_PHRASES) + r')\b'
    
    # Function to detect feature requests in text
    def has_feature_request(text):
        if not isinstance(text, str):
            return False
        
        # Check for feature request phrases
        if re.search(feature_pattern, text.lower()):
            return True
        
        return False
    
    # Apply function to content column
    reviews_df['is_feature_request'] = reviews_df['content'].apply(has_feature_request)
    
    return reviews_df

# Function to generate summary insights
def generate_insights(df):
    total_reviews = len(df)
    
    # Sentiment breakdown
    sentiment_counts = df['sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    
    positive_percent = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
    negative_percent = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Average rating
    avg_rating = df['score'].mean()
    
    # Feature requests
    feature_requests = df[df['is_feature_request']]
    feature_request_count = len(feature_requests)
    feature_percent = (feature_request_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Get top positive aspects (from positive reviews)
    positive_df = df[df['sentiment'] == 'Positive']
    # Get top negative aspects (from negative reviews)
    negative_df = df[df['sentiment'] == 'Negative']
    
    # Get top feature requests
    feature_texts = feature_requests['processed_text'].dropna().tolist()
    
    insights = {
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'positive_count': positive_count,
        'positive_percent': positive_percent,
        'negative_count': negative_count,
        'negative_percent': negative_percent,
        'feature_request_count': feature_request_count,
        'feature_percent': feature_percent,
        'positive_df': positive_df,
        'negative_df': negative_df,
        'feature_texts': feature_texts
    }
    
    return insights

# Function to extract keywords using TF-IDF
def extract_keywords(texts, n=10):
    if not texts or len(texts) == 0:
        return []
    
    # Combine texts into one large document
    combined_text = ' '.join(texts)
    
    # If text is too short, return empty list
    if len(combined_text.split()) < 3:
        return []
    
    # Define TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    try:
        # Transform text to TF-IDF feature matrix
        X = tfidf.fit_transform([combined_text])
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        scores = X.toarray()[0]
        
        # Create DataFrame with feature names and scores
        keywords_df = pd.DataFrame({
            'keyword': feature_names,
            'score': scores
        })
        
        # Sort by score and return top n
        keywords_df = keywords_df.sort_values('score', ascending=False).head(n)
        
        return keywords_df['keyword'].tolist()
    except:
        return []

# Function to generate a textual summary of insights
def generate_summary(insights):
    # Positive aspects
    positive_texts = insights['positive_df']['processed_text'].dropna().tolist()
    positive_keywords = extract_keywords(positive_texts)
    
    # Negative aspects
    negative_texts = insights['negative_df']['processed_text'].dropna().tolist()
    negative_keywords = extract_keywords(negative_texts)
    
    # Feature requests
    feature_keywords = extract_keywords(insights['feature_texts'])
    
    # Build summary
    summary = f"""
### App Review Summary

Based on {insights['total_reviews']} reviews, your app has an average rating of {insights['avg_rating']:.1f}/5.0.

**Sentiment Breakdown:**
- Positive: {insights['positive_count']} reviews ({insights['positive_percent']:.1f}%)
- Negative: {insights['negative_count']} reviews ({insights['negative_percent']:.1f}%)
- Feature Requests: {insights['feature_request_count']} reviews ({insights['feature_percent']:.1f}%)

**What Users Like:**
{', '.join(positive_keywords) if positive_keywords else 'Not enough data to determine'}

**What Users Dislike:**
{', '.join(negative_keywords) if negative_keywords else 'Not enough data to determine'}

**Feature Requests/Improvements:**
{', '.join(feature_keywords) if feature_keywords else 'No clear feature requests identified'}
"""
    
    return summary

# Main app logic
# Fetch + analysis results are kept in session_state so the dashboard survives
# reruns triggered by widget interactions (e.g. the Detailed Reviews filters).
if fetch_button and app_id:
    with st.spinner("Fetching reviews from Play Store..."):
        reviews_data = scrape_reviews(app_id, count, lang, country, sort_order)

    if not reviews_data:
        # Unknown app IDs return no reviews rather than raising, so say so explicitly
        # and drop the previous app's results so they aren't mistaken for this one
        st.session_state.pop('analysis', None)
        if reviews_data is not None:
            st.warning(
                f"No reviews found for **{app_id}** in {lang}/{country}. Check the app ID "
                "(e.g. com.example.app) or try another language/country."
            )
    else:
        # Save reviews locally
        json_file, csv_file = save_reviews(reviews_data, app_id)

        # Convert to DataFrame for analysis
        raw_df = pd.DataFrame(reviews_data)
        df = raw_df.copy()

        # Analyze sentiment
        df = analyze_sentiment(df)

        # Detect feature requests
        df = detect_feature_requests(df)

        st.session_state['analysis'] = {
            'app_id': app_id,
            'raw_df': raw_df,
            'df': df,
            'json_file': json_file,
            'csv_file': csv_file,
        }

if 'analysis' in st.session_state:
    analysis = st.session_state['analysis']
    raw_df = analysis['raw_df']
    df = analysis['df']

    st.success(f"Successfully retrieved {len(raw_df)} reviews for {analysis['app_id']}")
    st.success(f"Reviews saved to {analysis['json_file']} and {analysis['csv_file']}")

    # Show raw data
    with st.expander("Raw Data Preview"):
        st.dataframe(raw_df)

    # Generate insights
    insights = generate_insights(df)

    # Create a summary tab at the top
    st.header("📊 Analysis Dashboard")
    
    # Create tabs for different views
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Overview", "Sentiment Analysis", "Key Phrases", "Detailed Reviews"])
    
    with tab0:
        st.subheader("🧭 Executive Summary")
        exec_summary = build_executive_summary(df)
        st.markdown(format_headline(exec_summary))

        issues_col, likes_col = st.columns(2)
        for col, title, themes, empty_msg in [
            (issues_col, "🔴 Main Issues", exec_summary.issues, "No recurring issues found in 1–2★ reviews."),
            (likes_col, "🟢 What Users Like", exec_summary.likes, "No recurring themes found in 4–5★ reviews."),
        ]:
            with col:
                st.markdown(f"#### {title}")
                if not themes:
                    st.caption(empty_msg)
                for i, theme in enumerate(themes, 1):
                    st.markdown(
                        f"**{i}. {theme.phrase}** · {theme.review_count} reviews "
                        f"({theme.share:.0%}) · 👍 {theme.upvotes}"
                    )
                    st.caption(f"“{theme.quote}”")

        st.divider()
        st.subheader("📝 App Review Summary")
        summary = generate_summary(insights)
        st.markdown(summary)
        
        # Display most representative positive reviews
        if not insights['positive_df'].empty:
            st.subheader("Most Representative Positive Reviews")
            top_positive = insights['positive_df'].sort_values('sentiment_score', ascending=False).head(3)
            for _, row in top_positive.iterrows():
                st.markdown(f"""
                ⭐ **{row['score']}/5** - "{row['content']}"
                """)
        
        # Display most representative negative reviews
        if not insights['negative_df'].empty:
            st.subheader("Most Representative Negative Reviews")
            top_negative = insights['negative_df'].sort_values('sentiment_score').head(3)
            for _, row in top_negative.iterrows():
                st.markdown(f"""
                ⭐ **{row['score']}/5** - "{row['content']}"
                """)
        
        # Display feature requests
        feature_requests = df[df['is_feature_request']]
        if not feature_requests.empty:
            st.subheader("Feature Requests")
            for _, row in feature_requests.head(5).iterrows():
                st.markdown(f"""
                💡 "{row['content']}"
                """)
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            fig = px.histogram(df, x='score', nbins=5, 
                              labels={'score': 'Rating', 'count': 'Number of Reviews'},
                              color_discrete_sequence=['#3366CC'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Rating Over Time")
            df['at'] = pd.to_datetime(df['at'])
            df['date'] = df['at'].dt.date
            avg_rating = df.groupby('date')['score'].mean().reset_index()
            fig = px.line(avg_rating, x='date', y='score', 
                         labels={'score': 'Average Rating', 'date': 'Date'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(df))
        col2.metric("Average Rating", f"{df['score'].mean():.2f}")
        col3.metric("Most Common Rating", df['score'].mode()[0])
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                        hole=0.4, color='Sentiment',
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rating vs Sentiment")
            fig = px.box(df, x='sentiment', y='score', 
                        color='sentiment',
                        color_discrete_map=colors,
                        labels={'sentiment': 'Sentiment', 'score': 'Rating'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment score distribution
        st.subheader("Sentiment Score Distribution")
        fig = px.histogram(df, x='sentiment_score', nbins=20,
                          color_discrete_sequence=['#3366CC'],
                          labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Reviews'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Word Clouds")
        
        # Create tabs for different sentiment word clouds
        wc_tab1, wc_tab2, wc_tab3, wc_tab4 = st.tabs(["All", "Positive", "Neutral", "Negative"])
        
        with wc_tab1:
            wordcloud, top_words = extract_key_phrases(df)
            st.image(wordcloud.to_array(), caption="All Reviews Word Cloud")
            st.subheader("Top Words/Phrases")
            st.table(top_words.head(10))
        
        with wc_tab2:
            wordcloud, top_words = extract_key_phrases(df, 'Positive')
            st.image(wordcloud.to_array(), caption="Positive Reviews Word Cloud")
            st.subheader("Top Positive Words/Phrases")
            st.table(top_words.head(10))
        
        with wc_tab3:
            wordcloud, top_words = extract_key_phrases(df, 'Neutral') 
            st.image(wordcloud.to_array(), caption="Neutral Reviews Word Cloud")
            st.subheader("Top Neutral Words/Phrases")
            st.table(top_words.head(10))
        
        with wc_tab4:
            wordcloud, top_words = extract_key_phrases(df, 'Negative')
            st.image(wordcloud.to_array(), caption="Negative Reviews Word Cloud")
            st.subheader("Top Negative Words/Phrases")
            st.table(top_words.head(10))
    
    with tab4:
        st.subheader("Detailed Reviews")
        
        # Filter options
        sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
        rating_filter = st.multiselect("Filter by Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        
        filtered_df = df.copy()
        
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
        
        if rating_filter:
            filtered_df = filtered_df[filtered_df['score'].isin(rating_filter)]
        
        # Display detailed reviews
        for _, row in filtered_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.metric("Rating", row['score'])
                with col2:
                    st.write(f"**{row['userName']}** - {row['at']}")
                    st.write(row['content'])
                with col3:
                    sentiment = row['sentiment']
                    if sentiment == 'Positive':
                        st.markdown("😀 Positive")
                    elif sentiment == 'Neutral':
                        st.markdown("😐 Neutral")
                    else:
                        st.markdown("😞 Negative")
                st.divider()
else:
    st.info("Please enter your app ID and click 'Fetch Reviews' to start the analysis.")

# Add instructions
with st.sidebar:
    st.markdown("---")
    st.subheader("How to find your App ID")
    st.markdown("""
    1. Go to your app's Google Play Store page
    2. The App ID is in the URL: play.google.com/store/apps/details?id=**your.app.id**
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit")
