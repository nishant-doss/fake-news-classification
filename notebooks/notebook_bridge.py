"""
Notebook Bridge - Data Persistence Helper for All Notebooks
This module provides functions to save and load data between notebooks
"""

import pandas as pd
import pickle
import numpy as np
import os
import joblib
from datetime import datetime

def create_persistence_directory():
    """Create the persistence directory if it doesn't exist"""
    os.makedirs('../data/persistence', exist_ok=True)

def save_notebook_data(notebook_num, **kwargs):
    """
    Save data from a notebook with automatic organization
    
    Args:
        notebook_num (int): Notebook number (1-6)
        **kwargs: Variables to save
    """
    create_persistence_directory()
    
    print(f"💾 SAVING NOTEBOOK {notebook_num} DATA...")
    
    # Save all variables as pickle
    save_path = f'../data/persistence/notebook{notebook_num}_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)
    
    # Save DataFrames as CSV if present
    csv_count = 0
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            csv_path = f'../data/persistence/notebook{notebook_num}_{key}.csv'
            value.to_csv(csv_path, index=False)
            csv_count += 1
    
    # Save metadata
    metadata = {
        'notebook': notebook_num,
        'timestamp': datetime.now(),
        'variables_saved': list(kwargs.keys()),
        'csv_files_saved': csv_count
    }
    
    with open(f'../data/persistence/notebook{notebook_num}_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved {len(kwargs)} variables from Notebook {notebook_num}")
    print(f"📁 Main file: notebook{notebook_num}_data.pkl")
    print(f"📊 CSV files: {csv_count} DataFrames saved")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_notebook_data(notebook_num):
    """
    Load data from a specific notebook
    
    Args:
        notebook_num (int): Notebook number to load from
        
    Returns:
        dict: Dictionary containing all saved variables
    """
    print(f"📂 LOADING NOTEBOOK {notebook_num} DATA...")
    
    try:
        # Load main data
        data_path = f'../data/persistence/notebook{notebook_num}_data.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Load metadata
        metadata_path = f'../data/persistence/notebook{notebook_num}_metadata.pkl'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ Loaded {len(data)} variables from Notebook {notebook_num}")
            print(f"🕒 Data saved: {metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📋 Variables: {', '.join(metadata['variables_saved'][:5])}{'...' if len(metadata['variables_saved']) > 5 else ''}")
        
        return data
        
    except FileNotFoundError:
        print(f"❌ No data found for Notebook {notebook_num}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading Notebook {notebook_num} data: {e}")
        return None

def load_latest_data():
    """Load data from the most recent notebook"""
    print("🔍 SEARCHING FOR LATEST NOTEBOOK DATA...")
    
    # Check notebooks 6 to 1
    for notebook_num in range(6, 0, -1):
        data = load_notebook_data(notebook_num)
        if data is not None:
            print(f"📍 Using data from Notebook {notebook_num} (most recent)")
            return data, notebook_num
    
    print("❌ No notebook data found")
    return None, None

def load_all_results():
    """Load results from all notebooks for comparison"""
    print("📊 LOADING ALL NOTEBOOK RESULTS...")
    
    all_results = {}
    
    for notebook_num in range(1, 7):
        data = load_notebook_data(notebook_num)
        if data is not None:
            # Extract key results
            results = {}
            for key, value in data.items():
                if 'accuracy' in key or 'result' in key or 'score' in key:
                    results[key] = value
            
            if results:
                all_results[f'notebook_{notebook_num}'] = results
    
    return all_results

def create_progress_summary():
    """Create a summary of progress through all notebooks"""
    print("📈 CREATING PROGRESS SUMMARY...")
    
    summary = {
        'notebooks_completed': [],
        'datasets_processed': [],
        'models_trained': [],
        'best_accuracies': {}
    }
    
    for notebook_num in range(1, 7):
        metadata_path = f'../data/persistence/notebook{notebook_num}_metadata.pkl'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            summary['notebooks_completed'].append(notebook_num)
            
            # Try to load data for more details
            data = load_notebook_data(notebook_num)
            if data:
                # Look for accuracy metrics
                for key, value in data.items():
                    if 'accuracy' in key and isinstance(value, (int, float)):
                        summary['best_accuracies'][f'notebook_{notebook_num}_{key}'] = value
    
    return summary

# Quick access functions for specific notebooks

def save_exploration_data(df, text_stats, subject_analysis, **extras):
    """Save data from exploration notebook (1)"""
    save_notebook_data(1, 
                      df=df, 
                      text_stats=text_stats, 
                      subject_analysis=subject_analysis,
                      **extras)

def save_preprocessing_data(df, preprocessor, X_train, X_test, y_train, y_test, 
                           trained_models, results, **extras):
    """Save data from preprocessing notebook (2)"""
    save_notebook_data(2,
                      df=df,
                      preprocessor=preprocessor,
                      X_train=X_train,
                      X_test=X_test, 
                      y_train=y_train,
                      y_test=y_test,
                      trained_models=trained_models,
                      results=results,
                      **extras)

def save_cnn_data(models, accuracies, predictions, tokenizer=None, **extras):
    """Save data from CNN notebook (3)"""
    save_notebook_data(3,
                      models=models,
                      accuracies=accuracies,
                      predictions=predictions,
                      tokenizer=tokenizer,
                      **extras)

def save_lstm_data(models, accuracies, predictions, tokenizer=None, **extras):
    """Save data from LSTM notebook (4)"""
    save_notebook_data(4,
                      models=models,
                      accuracies=accuracies,
                      predictions=predictions,
                      tokenizer=tokenizer,
                      **extras)

def save_transformer_data(model, accuracy, predictions, tokenizer, **extras):
    """Save data from transformer notebook (5)"""
    save_notebook_data(5,
                      model=model,
                      accuracy=accuracy,
                      predictions=predictions,
                      tokenizer=tokenizer,
                      **extras)

def save_comparison_data(all_results, visualizations, poster_data, **extras):
    """Save data from comparison notebook (6)"""
    save_notebook_data(6,
                      all_results=all_results,
                      visualizations=visualizations,
                      poster_data=poster_data,
                      **extras)

# Usage examples:
"""
# In any notebook:
from notebook_bridge import save_notebook_data, load_notebook_data, load_latest_data

# Save current notebook's data
save_notebook_data(2, df=df, model=model, accuracy=accuracy)

# Load from specific notebook  
data = load_notebook_data(1)
if data:
    df = data['df']
    model = data['model']

# Load from most recent
data, notebook_num = load_latest_data()
"""