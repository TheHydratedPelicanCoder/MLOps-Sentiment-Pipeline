"""
Download and prepare IMDb dataset for sentiment analysis
"""
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    print("Downloading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # Convert to pandas
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Take subset for faster training (use full dataset in production)
    train_df = train_df.sample(n=5000, random_state=42)
    test_df = test_df.sample(n=1000, random_state=42)
    
    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    # Save reference data for drift detection (first 500 samples)
    train_df.head(500).to_csv('data/reference.csv', index=False)
    
    print(f"✅ Data prepared:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Reference samples: 500")

if __name__ == "__main__":
    prepare_data()