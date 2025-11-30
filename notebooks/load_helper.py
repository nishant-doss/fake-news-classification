"""
Load Helper - Safe loading functions for notebooks
Handles class imports and data loading issues
"""

import pandas as pd
import pickle
import numpy as np
import os
import sys
import joblib

# Add paths for custom classes
sys.path.append('../src')
sys.path.append('./src')
sys.path.append('.')

def safe_load_preprocessor():
    """Safely load the text preprocessor"""
    try:
        # First try to import the class
        from text_preprocessor import TextPreprocessor
        
        # Then try to load the saved preprocessor
        if os.path.exists('../models/text_preprocessor.pkl'):
            preprocessor = joblib.load('../models/text_preprocessor.pkl')
            print("✅ Loaded TextPreprocessor from ../models/text_preprocessor.pkl")
            return preprocessor
        else:
            print("⚠️ No saved preprocessor found, creating new one...")
            from text_preprocessor import ensure_nltk_data
            ensure_nltk_data()
            preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
            return preprocessor
            
    except ImportError:
        print("⚠️ TextPreprocessor class not found, creating inline...")
        
        # Define TextPreprocessor inline as fallback
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        
        class TextPreprocessor:
            def __init__(self, use_stemming=True, remove_stopwords=True):
                self.use_stemming = use_stemming
                self.remove_stopwords = remove_stopwords
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            
            def clean_text(self, text):
                text = text.lower()
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S+@\S+', '', text)
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = ' '.join(text.split())
                return text
            
            def tokenize_and_process(self, text):
                tokens = word_tokenize(text)
                if self.remove_stopwords:
                    tokens = [token for token in tokens if token not in self.stop_words]
                if self.use_stemming:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                else:
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                return ' '.join(tokens)
            
            def preprocess(self, text):
                if pd.isna(text):
                    return ''
                text = self.clean_text(text)
                text = self.tokenize_and_process(text)
                return text
        
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        print("✅ Created new TextPreprocessor instance")
        return preprocessor
    
    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
        return None

def load_notebook_data_safe(notebook_num):
    """Safely load notebook data with error handling"""
    try:
        # Try loading from persistence directory
        persistence_file = f'../data/persistence/notebook{notebook_num}_data.pkl'
        if os.path.exists(persistence_file):
            with open(persistence_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded data from notebook {notebook_num}")
            return data
        else:
            print(f"⚠️ No persistence file found for notebook {notebook_num}")
            return None
    except Exception as e:
        print(f"❌ Error loading notebook {notebook_num} data: {e}")
        return None

def load_train_test_data():
    """Load train/test splits with fallback options"""
    print("📊 Loading train/test data...")
    
    # Option 1: Load from persistence directory
    try:
        if os.path.exists('../data/persistence/notebook2_train_data.csv'):
            train_df = pd.read_csv('../data/persistence/notebook2_train_data.csv')
            test_df = pd.read_csv('../data/persistence/notebook2_test_data.csv')
            
            X_train = train_df['text']
            y_train = train_df['label']
            X_test = test_df['text'] 
            y_test = test_df['label']
            
            print(f"✅ Loaded from persistence: {len(X_train)} train, {len(X_test)} test")
            return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"⚠️ Error loading from persistence: {e}")
    
    # Option 2: Load processed dataset and split
    try:
        if os.path.exists('../data/persistence/notebook2_processed_df.csv'):
            df = pd.read_csv('../data/persistence/notebook2_processed_df.csv')
        elif os.path.exists('../data/combined_news_dataset.csv'):
            df = pd.read_csv('../data/combined_news_dataset.csv')
        else:
            print("❌ No dataset found")
            return None, None, None, None
        
        # If no processed_text column, need to preprocess
        if 'processed_text' not in df.columns:
            print("⚠️ No processed text found, basic preprocessing needed...")
            df['combined_text'] = df['title'] + ' ' + df['text']
            # Simple preprocessing
            df['processed_text'] = df['combined_text'].str.lower()
        
        # Split the data
        from sklearn.model_selection import train_test_split
        X = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Created new splits: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        return None, None, None, None

def setup_notebook_environment(notebook_name):
    """Set up environment for a notebook with all necessary imports and data"""
    print(f"🔧 Setting up environment for {notebook_name}...")
    
    # Basic imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load preprocessor
    preprocessor = safe_load_preprocessor()
    
    # Load train/test data
    X_train, X_test, y_train, y_test = load_train_test_data()
    
    # Load main dataset
    df = None
    if os.path.exists('../data/persistence/notebook2_processed_df.csv'):
        df = pd.read_csv('../data/persistence/notebook2_processed_df.csv')
    elif os.path.exists('../data/combined_news_dataset.csv'):
        df = pd.read_csv('../data/combined_news_dataset.csv')
    
    print("✅ Environment setup complete!")
    
    return {
        'df': df,
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# Quick usage function
def quick_load():
    """Quick load function for notebooks"""
    print("🚀 QUICK LOAD: Setting up notebook environment...")
    return setup_notebook_environment("Current Notebook")