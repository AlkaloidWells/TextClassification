import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
from helper.utils import TextCleaner, get_absolute_path
from sklearn.model_selection import StratifiedShuffleSplit

def train():
    # Load and prepare the e-commerce dataset
    data = pd.read_csv(get_absolute_path('DataSet/product_data.csv'))  # 'product_id', 'product_name', 'product_description', 'category_id', 'category_name'

    # Combine 'product_name' and 'product_description' for feature text
    data['text'] = data['product_name'] + ' ' + data['product_description']

    # Apply text cleaning
    data['text'] = TextCleaner().fit_transform(data['text'])

    # Filter out classes with fewer than 2 samples
    data = data.groupby('category_id').filter(lambda x: len(x) > 1)

    # Split data into features and labels
    X = data['text']
    y = data['category_id']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build a classification pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    # Cross-validation using StratifiedShuffleSplit
    stratified_shuffle = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=stratified_shuffle, scoring='accuracy')
    print("Stratified Shuffle Cross-validation scores:", cross_val_scores)
    print("Mean Stratified Shuffle Cross-validation accuracy:", cross_val_scores.mean())

    # Hyperparameter tuning with StratifiedKFold in GridSearchCV
    stratified_kf = StratifiedKFold(n_splits=3)  # Change to 3 to match the data
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kf, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    best_pipeline = grid_search.best_estimator_

    # Evaluate the tuned model on the test set
    y_pred = best_pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Tuned Model Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save the tuned model
    with open(get_absolute_path('models/fileLR/category_classifier_model.pkl'), 'wb') as model_file:
        pickle.dump(best_pipeline, model_file)

    # Save category mapping
    category_mapping = data[['category_id', 'category_name']].drop_duplicates().set_index('category_id').to_dict()['category_name']
    with open(get_absolute_path('models/fileLR/category_mapping.pkl'), 'wb') as mapping_file:
        pickle.dump(category_mapping, mapping_file)

    print("Model and category mapping saved successfully.")


