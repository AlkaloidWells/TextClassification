import Classifiers.textclassiferLR as cl1
import Classifiers.textclassiferLRS as cl2
import Classifiers.textclassiferCNN as cl3
import Classifiers.textclassiferXB as cl4
import Classifiers.textclassiferRF as cl5
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from helper.utils import TextCleaner


def get_best_predictions(query):
    # Get predictions from each classifier
    preds_lr = cl1.classify_query(query)
    preds_lrs = cl2.classify_query(query)
    preds_cnn = cl3.classify_query(query)
    preds_xb = cl4.classify_query(query)
    preds_rf = cl5.classify_query(query)

    # Combine all predictions into a single DataFrame
    all_preds = pd.concat([preds_lr, preds_lrs, preds_cnn, preds_xb, preds_rf], ignore_index=True)

    # Calculate the frequency of each category ID
    freq = all_preds['category_id'].value_counts().reset_index()
    freq.columns = ['category_id', 'frequency']

    # Merge the frequency with the original predictions
    all_preds = all_preds.merge(freq, on='category_id')

    # Calculate a score based on F1 score and frequency
    all_preds['score'] = all_preds['probability'] * all_preds['frequency']

    # Sort by the calculated score and get the top 2 predictions
    best_preds = all_preds.sort_values(by='score', ascending=False).head(2)

    return best_preds.reset_index(drop=True)



def get_voting_classifier_predictions(query):
    # Define the classifiers
    classifiers = [
        ('lr', cl1.classify_query),
        ('lrs', cl2.classify_query),
        ('cnn', cl3.classify_query),
        ('xb', cl4.classify_query),
        ('rf', cl5.classify_query)
    ]

    # Create a VotingClassifier
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

    # Create a pipeline with TF-IDF vectorizer and VotingClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('voting_clf', voting_clf)
    ])

    # Clean the query
    cleaned_query = TextCleaner().fit_transform([query])

    # Transform the query using the TF-IDF vectorizer
    query_tfidf = pipeline.named_steps['tfidf'].transform(cleaned_query)

    # Predict probabilities for each category
    probabilities = voting_clf.predict_proba(query_tfidf)[0]

    # Create a DataFrame for categories and their probabilities
    categories_df = pd.DataFrame({
        'category_id': voting_clf.classes_,
        'probability': probabilities
    })

    # Sort by probability and get the top 2 categories
    top_categories = categories_df.sort_values(by='probability', ascending=False).head(2)

    return top_categories.reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    query = "iphone iphone gb fairly used but works perfectly battery life at"
    
    # Custom voting mechanism
    best_predictions_custom = get_best_predictions(query)
    print(f"The best predictions for the query '{query}' using custom voting are:\n{best_predictions_custom}")
    
    # VotingClassifier
    best_predictions_voting = get_voting_classifier_predictions(query)
    print(f"The best predictions for the query '{query}' using VotingClassifier are:\n{best_predictions_voting}")