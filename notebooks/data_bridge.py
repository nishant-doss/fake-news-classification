# Data Bridge - Helper functions to share data between notebooks

import pandas as pd
import pickle
import numpy as np
import os

def save_preprocessing_data(df, preprocessor, X_train, X_test, y_train, y_test, tokenizer=None):
    """Save all preprocessing outputs for use in subsequent notebooks"""
    
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../data', exist_ok=True)
    
    # Save main dataframe
    df.to_csv('../data/processed_dataset.csv', index=False)
    
    # Save train/test splits
    if hasattr(X_train, 'toarray'):  # If sparse matrix
        X_train_df = pd.DataFrame(X_train.toarray())
        X_test_df = pd.DataFrame(X_test.toarray())
    else:
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
    
    X_train_df.to_csv('../data/X_train.csv', index=False)
    X_test_df.to_csv('../data/X_test.csv', index=False)
    
    pd.Series(y_train).to_csv('../data/y_train.csv', index=False)
    pd.Series(y_test).to_csv('../data/y_test.csv', index=False)
    
    # Save objects
    with open('../models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    if tokenizer:
        with open('../models/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
    
    print("✅ Data saved successfully!")
    print("Files saved:")
    print("- ../data/processed_dataset.csv")
    print("- ../data/X_train.csv, X_test.csv")
    print("- ../data/y_train.csv, y_test.csv") 
    print("- ../models/preprocessor.pkl")
    if tokenizer:
        print("- ../models/tokenizer.pkl")

def load_preprocessing_data():
    """Load all preprocessing outputs"""
    
    # Load main dataframe
    df = pd.read_csv('../data/processed_dataset.csv')
    
    # Load train/test splits
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('../data/y_test.csv').iloc[:, 0]
    
    # Load objects
    with open('../models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    tokenizer = None
    if os.path.exists('../models/tokenizer.pkl'):
        with open('../models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    
    print("✅ Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    if tokenizer:
        return df, preprocessor, X_train, X_test, y_train, y_test, tokenizer
    else:
        return df, preprocessor, X_train, X_test, y_train, y_test

def save_model_results(model, model_name, accuracy, predictions=None):
    """Save trained model and results"""
    
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Save model
    with open(f'../models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'timestamp': pd.Timestamp.now()
    }
    
    results_df = pd.DataFrame([results])
    
    # Append to results file
    results_file = '../results/model_results.csv'
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    
    results_df.to_csv(results_file, index=False)
    
    if predictions is not None:
        pred_df = pd.DataFrame({
            'predictions': predictions,
            'model': model_name
        })
        pred_df.to_csv(f'../results/{model_name}_predictions.csv', index=False)
    
    print(f"✅ Model {model_name} saved successfully!")
    print(f"Accuracy: {accuracy:.4f}")

# Usage examples:
"""
# At the end of notebook 2:
save_preprocessing_data(df, preprocessor, X_train, X_test, y_train, y_test)

# At the start of notebook 3:
df, preprocessor, X_train, X_test, y_train, y_test = load_preprocessing_data()

# After training a model:
save_model_results(cnn_model, 'cnn_standard', accuracy, predictions)
"""