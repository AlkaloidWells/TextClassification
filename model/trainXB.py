import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.over_sampling import SMOTE
import pickle
from model.utils import TextCleaner, get_absolute_path
from xgboost import XGBClassifier
import os

# Suppress cuBLAS warnings
os.environ['XGBOOST_ENABLE_CUDA'] = '0'

def train():
    # Load and preprocess data
    data = pd.read_csv(get_absolute_path('product_data.csv'))
    data['text'] = data['product_name'] + ' ' + data['product_description']
    data['text'] = TextCleaner().fit_transform(data['text'])

    # Map category_id to category_name
    category_mapping = dict(zip(data['category_id'], data['category_name']))  # Assuming category_name exists

    # Remove classes with fewer than 3 samples
    min_samples_per_class = 3
    data = data.groupby('category_id').filter(lambda x: len(x) >= min_samples_per_class)

    print("Class distribution after filtering:")
    print(data['category_id'].value_counts())

    # Encode category names
    label_encoder = LabelEncoder()
    data['encoded_category'] = label_encoder.fit_transform(data['category_name'])

    X = data['text']
    y = data['encoded_category']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create pipeline with SMOTE and XGBoost
    pipeline = IMBPipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Print evaluation metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save the pipeline, label encoder, and category mapping
    with open(get_absolute_path('model/pretrained/file3/category_classifier_model.pkl'), 'wb') as model_file:
        pickle.dump(pipeline, model_file)

    with open(get_absolute_path('model/pretrained/file3/label_encoder.pkl'), 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

    with open(get_absolute_path('model/pretrained/file3/category_mapping.pkl'), 'wb') as mapping_file:
        pickle.dump(category_mapping, mapping_file)

    print("Model, encoder, and mappings saved successfully.")
