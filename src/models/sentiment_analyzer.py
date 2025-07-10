# src/models/sentiment_analyzer.py

import pandas as pd
import nltk

class AspectSentimentAnalyzer:
    def __init__(self, aspects: list[str]):
        """
        Initializes the AspectSentimentAnalyzer class with a list of aspects to target.
        :param aspects: list of aspects to target
        """
        self.aspects = aspects