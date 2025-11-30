# QUICK SETUP CELL - Run this at the beginning of any notebook
# This cell handles all imports and data loading safely

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('../src')
sys.path.append('.')

print("🔧 SETTING UP NOTEBOOK ENVIRONMENT...")

# Function to safely load preprocessor
def load_preprocessor_safe():
    try:
        # Try to load from persistence
        if os.path.exists('../data/persistence/notebook2_preprocessor.pkl'):
            with open('../data/persistence/notebook2_preprocessor.pkl', 'rb') as f:
                # Define TextPreprocessor class first
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
                
                preprocessor = pickle.load(f)
                print("✅ Loaded saved TextPreprocessor")
                return preprocessor
        
        # Create new preprocessor if loading fails
        print("⚠️ Creating new TextPreprocessor...")
        from text_preprocessor import TextPreprocessor, ensure_nltk_data
        ensure_nltk_data()
        return TextPreprocessor(use_stemming=True, remove_stopwords=True)
        
    except Exception as e:
        print(f"❌ Error with preprocessor: {e}")
        return None

# Function to load train/test data
def load_data_safe():
    try:
        # Try loading from persistence
        if os.path.exists('../data/persistence/notebook2_train_data.csv'):
            train_df = pd.read_csv('../data/persistence/notebook2_train_data.csv')
            test_df = pd.read_csv('../data/persistence/notebook2_test_data.csv')
            
            X_train = train_df['text']
            y_train = train_df['label']
            X_test = test_df['text']
            y_test = test_df['label']
            
            print(f"✅ Loaded splits: {len(X_train)} train, {len(X_test)} test")
            return X_train, X_test, y_train, y_test
            
    except Exception as e:
        print(f"⚠️ Error loading splits: {e}")
    
    # Fallback: load main dataset and split
    try:
        if os.path.exists('../data/persistence/notebook2_processed_df.csv'):
            df = pd.read_csv('../data/persistence/notebook2_processed_df.csv')
        else:
            df = pd.read_csv('../data/combined_news_dataset.csv')
            df['combined_text'] = df['title'] + ' ' + df['text']
            df['processed_text'] = df['combined_text'].str.lower()  # Simple preprocessing
        
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

# Load main dataset
try:
    if os.path.exists('../data/persistence/notebook2_processed_df.csv'):
        df = pd.read_csv('../data/persistence/notebook2_processed_df.csv')
        print(f"✅ Loaded processed dataset: {df.shape}")
    else:
        df = pd.read_csv('../data/combined_news_dataset.csv')
        print(f"✅ Loaded raw dataset: {df.shape}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None

# Load preprocessor
preprocessor = load_preprocessor_safe()

# Load train/test data
X_train, X_test, y_train, y_test = load_data_safe()

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("🚀 NOTEBOOK ENVIRONMENT READY!")
print(f"📊 Dataset: {df.shape if df is not None else 'Not loaded'}")
print(f"🔀 Data splits: Train={len(X_train) if X_train is not None else 0}, Test={len(X_test) if X_test is not None else 0}")
print(f"⚙️ Preprocessor: {'✅ Ready' if preprocessor else '❌ Failed'}")
print("\n🔗 All variables loaded: df, X_train, X_test, y_train, y_test, preprocessor")