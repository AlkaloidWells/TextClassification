import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import RandomOverSampler
import pickle
from model.utils import TextCleaner, get_absolute_path
from xgboost import XGBClassifier
import numpy as np

def train():
    # Load and prepare the e-commerce dataset
    data = pd.read_csv(get_absolute_path('product_data.csv'))

    # Combine 'product_name' and 'product_description' for feature text
    data['text'] = data['product_name'] + ' ' + data['product_description']

    # Apply text cleaning
    data['text'] = TextCleaner().fit_transform(data['text'])

    # Filter out classes with very few samples (less than 5)
    min_samples_per_class = 5
    data = data.groupby('category_id').filter(lambda x: len(x) >= min_samples_per_class)

    # Print class distribution before balancing
    print("Class distribution before balancing:")
    print(data['category_id'].value_counts())

    # Split data into features and labels
    X = data['text']
    y = data['category_id']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate sampling strategy
    sampling_strategy = 'not majority'

    # Build a classification pipeline with RandomOverSampler
    pipeline = IMBPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('sampler', RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])

    # Hyperparameter tuning with StratifiedKFold
    stratified_kf = StratifiedKFold(n_splits=3)
    param_grid = {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.2]
    }

    best_score = -1
    best_params = None
    best_pipeline = None

    for n_estimators in param_grid['clf__n_estimators']:
        for max_depth in param_grid['clf__max_depth']:
            for learning_rate in param_grid['clf__learning_rate']:
                pipeline.set_params(clf__n_estimators=n_estimators, clf__max_depth=max_depth, clf__learning_rate=learning_rate)
                fold_scores = []

                for train_idx, val_idx in stratified_kf.split(X_train, y_train):
                    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    pipeline.fit(X_train_fold, y_train_fold)
                    score = pipeline.score(X_val_fold, y_val_fold)
                    fold_scores.append(score)

                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate
                    }
                    best_pipeline = pipeline

    print("Best parameters found:", best_params)

    # Train final model with best parameters
    best_pipeline.fit(X_train, y_train)

    # Print class distribution after sampling (for training data)
    tfidf_train = best_pipeline.named_steps['tfidf'].transform(X_train)
    _, y_resampled = best_pipeline.named_steps['sampler'].fit_resample(tfidf_train, y_train)
    print("\nClass distribution after sampling (training data):")
    print(pd.Series(y_resampled).value_counts())

    # Evaluate the tuned model on the test set
    y_pred = best_pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Tuned Model Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save the tuned model
    with open(get_absolute_path('model/pretrained/file2/category_classifier_model.pkl'), 'wb') as model_file:
        pickle.dump(best_pipeline, model_file)

    # Save category mapping
    category_mapping = data[['category_id', 'category_name']].drop_duplicates().set_index('category_id').to_dict()['category_name']
    with open(get_absolute_path('model/pretrained/file2/category_mapping.pkl'), 'wb') as mapping_file:
        pickle.dump(category_mapping, mapping_file)

    print("Model and category mapping saved successfully.")
