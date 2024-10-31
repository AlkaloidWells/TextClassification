from sklearn.base import BaseEstimator, TransformerMixin
import re
import os
from spellchecker import SpellChecker


# Text cleaning transformer
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(lambda x: self.clean_text(x))
    
    @staticmethod
    def clean_text(text):
        text = text.lower()  # Lowercase text
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    
    
def get_absolute_path(file_path):
    return os.path.abspath(file_path)


# In model/utils.py

# In model/utils.py

from spellchecker import SpellChecker  # Make sure you import your spell checker

class SpellCheckerPipeline:
    def __init__(self):
        # Initialize your spell checker here
        self.spell = SpellChecker()  # Replace with your actual spell checker initialization

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Replace None values with an empty string before spell correction
        X = [text if text is not None else "" for text in X]
        return [" ".join([self.spell.correction(word) for word in text.split()]) for text in X]


