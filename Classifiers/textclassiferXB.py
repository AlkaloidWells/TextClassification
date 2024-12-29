import pickle
import pandas as pd
from helper.utils import get_absolute_path, TextCleaner

def classify_query(query): 
    model_file = get_absolute_path('models/fileXB/category_classifier_model.pkl')
    mapping_file = get_absolute_path('models/fileXB/label_encoder.pkl')
    category_mapping_file = get_absolute_path('models/fileXB/category_mapping.pkl')
    tfidf_file = get_absolute_path('models/fileXB/tfidf_vectorizer.pkl')

    # Load the model
    with open(model_file, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Load the LabelEncoder
    with open(mapping_file, 'rb') as mapping_file:
        label_encoder = pickle.load(mapping_file)

    # Load the category mapping
    with open(category_mapping_file, 'rb') as category_mapping_file:
        category_mapping = pickle.load(category_mapping_file)

    # Load the TF-IDF vectorizer
    with open(tfidf_file, 'rb') as tfidf_file:
        tfidf_vectorizer = pickle.load(tfidf_file)

    # Clean the query
    cleaned_query = TextCleaner().fit_transform([query])

    # Transform the query using the TF-IDF vectorizer
    query_tfidf = tfidf_vectorizer.transform(cleaned_query)

    # Predict probabilities for each category
    probabilities = loaded_model.predict_proba(query_tfidf)[0]

    # Create a DataFrame for categories and their probabilities
    categories_df = pd.DataFrame({
        'category_id': loaded_model.classes_,
        'probability': probabilities
    })

    # Map encoded category ID to descriptive category names
    categories_df['category_name'] = categories_df['category_id'].map(lambda x: label_encoder.inverse_transform([x])[0])

    # Sort by probability and get the top 2 categories
    top_categories = categories_df.sort_values(by='probability', ascending=False).head(2)

    return top_categories.reset_index(drop=True)
