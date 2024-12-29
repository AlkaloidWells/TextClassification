import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from helper.utils import TextCleaner, get_absolute_path

def train():
    # Load and prepare the e-commerce dataset
    data = pd.read_csv(get_absolute_path('DataSet/product_data.csv'))
    
    # Combine 'product_name' and 'product_description' for feature text
    data['text'] = data['product_name'] + ' ' + data['product_description']
    
    # Apply text cleaning
    cleaner = TextCleaner()
    data['text'] = cleaner.fit_transform(data['text'])
    
    # Filter out classes with fewer than 5 samples
    class_counts = data['category_id'].value_counts()
    valid_categories = class_counts[class_counts >= 5].index
    data = data[data['category_id'].isin(valid_categories)]
    
    print("Class distribution before balancing:")
    print(data['category_id'].value_counts())
    
    # Prepare features and labels
    X = data['text']
    y = data['category_id']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Tokenize text
    max_words = 20000  # Increased vocabulary size
    max_len = 100     # Reduced sequence length
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Get minimum class size
    min_class_size = min(Counter(y_train).values())
    k_neighbors = min(min_class_size - 1, 5)
    
    print(f"Using k_neighbors = {k_neighbors} for SMOTE")
    
    # Apply SMOTE with adjusted parameters
    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy='auto'
    )
    
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(
            X_train_pad, y_train
        )
        print("SMOTE balancing successful")
        print("Class distribution after balancing:")
        print(pd.Series(y_train_balanced).value_counts())
        
    except ValueError as e:
        print(f"SMOTE failed: {str(e)}")
        print("Proceeding with unbalanced data")
        X_train_balanced, y_train_balanced = X_train_pad, y_train
    
    # Build improved CNN model
    n_classes = len(np.unique(y))
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        
        Conv1D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        
        Conv1D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        
        GlobalMaxPooling1D(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        X_train_balanced,
        y_train_balanced,
        epochs=20,  # Increased epochs since we have early stopping
        batch_size=64,  # Increased batch size
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the model with proper extension
    model.save(get_absolute_path('models/fileCNN/cnn_classifier_model.keras'))
    
    # Save tokenizer
    with open(get_absolute_path('models/fileCNN/tokenizer.pkl'), 'wb') as tok_file:
        pickle.dump(tokenizer, tok_file)
    
    # Save label encoder
    with open(get_absolute_path('models/fileCNN/label_encoder.pkl'), 'wb') as le_file:
        pickle.dump(label_encoder, le_file)
    
    # Save category mapping
    category_mapping = data[['category_id', 'category_name']].drop_duplicates().set_index('category_id').to_dict()['category_name']
    with open(get_absolute_path('models/fileCNN/category_mapping.pkl'), 'wb') as mapping_file:
        pickle.dump(category_mapping, mapping_file)
    
    print("Model and all components saved successfully.")