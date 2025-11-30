"""
Text Preprocessor Class
Shared across all notebooks for consistent text preprocessing
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_process(self, text):
        """Tokenize and apply stemming/lemmatization"""
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        if pd.isna(text):
            return ''
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize and process
        text = self.tokenize_and_process(text)
        
        return text

def safe_nltk_download(resource, subdir=None):
    """Download NLTK resources only if missing."""
    try:
        if subdir:
            nltk.data.find(f"{subdir}/{resource}")
        else:
            nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

# Ensure NLTK data is available
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    safe_nltk_download('punkt', 'tokenizers')
    safe_nltk_download('punkt_tab', 'tokenizers')
    safe_nltk_download('stopwords', 'corpora')
    safe_nltk_download('wordnet', 'corpora')
    safe_nltk_download('omw-1.4', 'corpora')