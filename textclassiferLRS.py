import pickle
import pandas as pd
from model.utils import get_absolute_path, TextCleaner, SpellCheckerPipeline

def classify_query(query): 
    model_file = get_absolute_path('model/pretrained/file2/category_classifier_model.pkl')
    mapping_file = get_absolute_path('model/pretrained/file2/category_mapping.pkl')

    # Load the model
    with open(model_file, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Load the category mapping
    with open(mapping_file, 'rb') as mapping_file:
        category_mapping = pickle.load(mapping_file)
    
    # Clean the new query
    cleaned_query = TextCleaner.clean_text(query)
    
    # Predict probabilities for each category
    probabilities = loaded_model.predict_proba([cleaned_query])[0]

    # Create a DataFrame for categories and their probabilities
    categories_df = pd.DataFrame({
        'category_id': loaded_model.classes_,
        'probability': probabilities
    })

    # Map category IDs to names
    categories_df['category_name'] = categories_df['category_id'].map(category_mapping)

    # Sort by probability and get the top 2 categories
    top_categories = categories_df.sort_values(by='probability', ascending=False).head(2)

    return top_categories.reset_index(drop=True)  # Return a DataFrame with top 2 categories

# Example usage
# if __name__ == "__main__":
#     query = "Example query here"
#     top_categories = classify_query(query)
#     print(top_categories)
