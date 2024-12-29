import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE
import pickle
from helper.utils import TextCleaner, get_absolute_path
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder

def train():
    # Load and prepare the e-commerce dataset
    print("Loading dataset...")
    data = pd.read_csv(get_absolute_path('DataSet/product_data.csv'))

    # Combine 'product_name' and 'product_description' for feature text
    print("Combining product_name and product_description...")
    data['text'] = data['product_name'] + ' ' + data['product_description']

    # Apply text cleaning
    print("Applying text cleaning...")
    data['text'] = TextCleaner().fit_transform(data['text'])

    # Filter out classes with very few samples (less than 6)
    min_samples_per_class = 6
    print(f"Filtering out classes with fewer than {min_samples_per_class} samples...")
    data = data.groupby('category_id').filter(lambda x: len(x) >= min_samples_per_class)

    # Print class distribution before balancing
    print("Class distribution before balancing:")
    print(data['category_id'].value_counts())

    # Encode category_id to numeric values
    print("Encoding category_id to numeric values...")
    label_encoder = LabelEncoder()
    data['category_id_encoded'] = label_encoder.fit_transform(data['category_id'])

    # Save the label mapping for later use
    print("Saving the label mapping...")
    category_mapping = dict(zip(data['category_id_encoded'], data['category_id']))

    print("Preprocessing complete. Proceeding to model training...")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category_id_encoded'], test_size=0.2, random_state=42, stratify=data['category_id_encoded'])

    # Define the pipeline
    pipeline = IMBPipeline([
        ('tfidf', TfidfVectorizer()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])

    # Define the parameter grid with fewer combinations
    param_grid = {
        'tfidf__max_features': [1000],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1]
    }

    # Perform grid search with cross-validation
    print("Performing grid search with cross-validation...")
    best_score = 0
    best_params = None
    best_pipeline = None

    for max_features in param_grid['tfidf__max_features']:
        for n_estimators in param_grid['classifier__n_estimators']:
            for learning_rate in param_grid['classifier__learning_rate']:
                print(f"Training with max_features={max_features}, n_estimators={n_estimators}, learning_rate={learning_rate}...")
                pipeline.set_params(tfidf__max_features=max_features, classifier__n_estimators=n_estimators, classifier__learning_rate=learning_rate)
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []

                for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
                    print(f"Starting fold {fold + 1}...")
                    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                    # Adjust k_neighbors for SMOTE based on the minimum class size in the fold
                    min_class_size = min(y_train_fold.value_counts())
                    k_neighbors = min(min_class_size - 1, 5)
                    pipeline.set_params(smote__k_neighbors=k_neighbors)

                    pipeline.fit(X_train_fold, y_train_fold)
                    score = pipeline.score(X_val_fold, y_val_fold)
                    scores.append(score)
                    print(f"Fold {fold + 1} score: {score}")

                mean_score = np.mean(scores)
                print(f"Mean cross-validation score: {mean_score}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'max_features': max_features,
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate
                    }
                    best_pipeline = pipeline

    print("Best parameters found:", best_params)

    # Train final model with best parameters
    print("Training final model with best parameters...")
    best_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Print class distribution after sampling (for training data)
    print("Transforming training data with TF-IDF...")
    tfidf_train = best_pipeline.named_steps['tfidf'].transform(X_train)
    print("Applying SMOTE to training data...")
    _, y_resampled = best_pipeline.named_steps['smote'].fit_resample(tfidf_train, y_train)
    print("\nClass distribution after sampling (training data):")
    print(pd.Series(y_resampled).value_counts())

    # Evaluate the tuned model on the test set
    print("Evaluating the tuned model on the test set...")
    y_pred = best_pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Tuned Model Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save the model, label encoder, category mapping, and TF-IDF vectorizer
    print("Saving the model, label encoder, category mapping, and TF-IDF vectorizer...")
    with open(get_absolute_path('models/fileXB/category_classifier_model.pkl'), 'wb') as model_file:
        pickle.dump(best_pipeline, model_file)
    with open(get_absolute_path('models/fileXB/label_encoder.pkl'), 'wb') as mapping_file:
        pickle.dump(label_encoder, mapping_file)
    with open(get_absolute_path('models/fileXB/category_mapping.pkl'), 'wb') as category_mapping_file:
        pickle.dump(category_mapping, category_mapping_file)
    with open(get_absolute_path('models/fileXB/tfidf_vectorizer.pkl'), 'wb') as tfidf_file:
        pickle.dump(best_pipeline.named_steps['tfidf'], tfidf_file)
