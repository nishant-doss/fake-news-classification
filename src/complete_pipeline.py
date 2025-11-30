#!/usr/bin/env python3
"""
Complete Fake News Classification Pipeline
Runs all steps in sequence with shared variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def main():
    print("🚀 Starting Complete Fake News Classification Pipeline")
    
    # Step 1: Load Data
    print("\n📊 Step 1: Loading Data...")
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset loaded: {len(df)} articles")
    print(f"Fake: {(df['label']==0).sum()}, True: {(df['label']==1).sum()}")
    
    # Step 2: Preprocessing
    print("\n🔧 Step 2: Preprocessing...")
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    # Simple preprocessing for now
    def simple_preprocess(text):
        return str(text).lower()
    
    df['processed_text'] = df['combined_text'].apply(simple_preprocess)
    
    # Step 3: Train-Test Split
    print("\n✂️ Step 3: Train-Test Split...")
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Step 4: Baseline Model
    print("\n🤖 Step 4: Training Baseline Model...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Step 5: Evaluation
    print("\n📈 Step 5: Evaluation...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
    
    # Step 6: Save Results
    print("\n💾 Step 6: Saving Models...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save processed data
    df.to_csv('data/processed_complete.csv', index=False)
    
    print("✅ Pipeline completed successfully!")
    print(f"📁 Models saved in: models/")
    print(f"📁 Data saved in: data/processed_complete.csv")
    
    return df, model, vectorizer, accuracy

if __name__ == "__main__":
    df, model, vectorizer, accuracy = main()