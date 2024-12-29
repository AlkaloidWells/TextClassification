import pickle
import pandas as pd
from helper.utils import get_absolute_path, TextCleaner
from sklearn.feature_extraction.text import TfidfVectorizer

def classify_query(query): 
    model_file = get_absolute_path('models/fileRF/category_classifier_model.pkl')
    mapping_file = get_absolute_path('models/fileRF/file4/label_encoder.pkl')
    category_mapping_file = get_absolute_path('models/fileRF/category_mapping.pkl')
    tfidf_file = get_absolute_path('models/fileRF/tfidf_vectorizer.pkl')

    # Load the model
    with open(model_file, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Load the label encoder
    with open(mapping_file, 'rb') as mapping_file:
        label_encoder = pickle.load(mapping_file)

    # Load the category mapping
    with open(category_mapping_file, 'rb') as category_mapping_file:
        category_mapping = pickle.load(category_mapping_file)

    # Load the TF-IDF vectorizer
    with open(tfidf_file, 'rb') as tfidf_file:
        tfidf_vectorizer = pickle.load(tfidf_file)

    # Preprocess the query
    cleaned_query = TextCleaner().fit_transform([query])
    query_tfidf = tfidf_vectorizer.transform(cleaned_query)

    # Predict the category
    predicted_category_encoded = loaded_model.predict(query_tfidf)
    predicted_category = label_encoder.inverse_transform(predicted_category_encoded)

    # Map the predicted category to the original category name
    predicted_category_name = category_mapping[predicted_category[0]]

    return predicted_category_name
