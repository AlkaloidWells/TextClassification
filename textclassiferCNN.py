import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.utils import get_absolute_path, TextCleaner

def classify_query(query):
    """
    Classifies a product query using the trained CNN model.
    
    Args:
        query (str): The product query text to classify
        
    Returns:
        pd.DataFrame: Top 2 predicted categories with their probabilities
    """
    # Load all necessary components
    model = load_model(get_absolute_path('model/pretrained/cnn_classifier_model.keras'))
    
    with open(get_absolute_path('model/pretrained/tokenizer.pkl'), 'rb') as tok_file:
        tokenizer = pickle.load(tok_file)
        
    with open(get_absolute_path('model/pretrained/label_encoder.pkl'), 'rb') as le_file:
        label_encoder = pickle.load(le_file)
        
    with open(get_absolute_path('model/pretrained/category_mapping.pkl'), 'rb') as mapping_file:
        category_mapping = pickle.load(mapping_file)
    
    # Clean the query - convert to pandas Series first
    cleaner = TextCleaner()
    query_series = pd.Series([query])
    cleaned_query = cleaner.transform(query_series)[0]
    
    # Convert to sequence and pad
    query_seq = tokenizer.texts_to_sequences([cleaned_query])
    query_pad = pad_sequences(query_seq, maxlen=100)  # Using same maxlen=100 as in training
    
    # Get probabilities
    probabilities = model.predict(query_pad, verbose=0)[0]
    
    # Create DataFrame with predictions
    categories_df = pd.DataFrame({
        'category_id': label_encoder.inverse_transform(range(len(probabilities))),
        'probability': probabilities
    })
    
    # Map category IDs to names
    categories_df['category_name'] = categories_df['category_id'].map(category_mapping)
    
    # Sort by probability and get top 2 categories
    top_categories = (
        categories_df.sort_values(by='probability', ascending=False)
                    .head(2)
                    .reset_index(drop=True)
    )
    
    return top_categories