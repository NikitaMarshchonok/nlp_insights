# src/sentiment.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize analyzer
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(texts: List[str]) -> List[float]:
    """
    Compute VADER compound sentiment scores for a list of texts.

    :param texts: list of raw or cleaned texts
    :return: list of compound sentiment scores (-1 to 1)
    """
    return [sia.polarity_scores(text)['compound'] for text in texts]


def label_sentiment(compound_scores: List[float], threshold: float = 0.0) -> List[str]:
    """
    Label texts as 'positive' or 'negative' based on compound score threshold.

    :param compound_scores: list of scores from analyze_sentiment
    :param threshold: boundary to classify positive vs negative
    :return: list of labels ('positive' or 'negative')
    """
    return ['positive' if score >= threshold else 'negative' for score in compound_scores]
