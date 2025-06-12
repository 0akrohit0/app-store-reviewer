# Play Store Reviews Analyzer

A Streamlit dashboard to scrape, store, and analyze Play Store reviews for your app. Get insights on sentiment, top phrases, and user feedback.

## Features

- Scrape reviews from Google Play Store using your app ID
- Store reviews in JSON and CSV format locally
- Perform sentiment analysis on reviews
- Generate visualizations:
  - Rating distribution
  - Sentiment breakdown
  - Word clouds for positive, neutral, and negative reviews
  - Top phrases analysis
  - Detailed review filtering

## Setup Instructions

1. Clone this repository or download the files
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your Play Store App ID (e.g., com.example.app)
2. Configure the number of reviews to fetch and other options
3. Click "Fetch Reviews" to start the analysis
4. Explore the dashboard tabs to gain insights:
   - Overview: General statistics and rating trends
   - Sentiment Analysis: Positive/neutral/negative breakdown
   - Key Phrases: Word clouds and top phrases 
   - Detailed Reviews: Filter and browse individual reviews

## Finding Your App ID

1. Go to your app's Google Play Store page
2. The App ID is in the URL: play.google.com/store/apps/details?id=**your.app.id**

## Requirements

See the `requirements.txt` file for a complete list of dependencies.

## Data Storage

Reviews are saved in the `data` directory in both JSON and CSV formats.
The filenames include the app ID and timestamp for easy identification. 