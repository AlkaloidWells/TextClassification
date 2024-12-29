import os
import Trainers.trainLR as trl
import Trainers.trainCNN as tcnn
import Trainers.trainLRS as tlrs
import Trainers.trainXB as txb
import Trainers.trainRF as trf
from helper.utils import get_absolute_path

def run_train_if_needed(train_function, model_dir):
    model_dir_path = get_absolute_path(model_dir)
    
    # Check if the directory exists and is not empty
    if not os.path.exists(model_dir_path) or not os.listdir(model_dir_path):
        # Create the directory if it doesn't exist
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        
        # Run the training function
        print(f"Training model and saving to {model_dir_path}...")
        train_function()
    else:
        print(f"Model directory {model_dir_path} already exists and is not empty. Skipping training.")

def run_train_lr():
    print("Checking and training Logistic Regression model if needed...")
    run_train_if_needed(trl.train, 'models/fileLR')

def run_train_cnn():
    print("Checking and training CNN model if needed...")
    run_train_if_needed(tcnn.train, 'models/fileCNN')

def run_train_lrs():
    print("Checking and training Logistic Regression with SMOTE model if needed...")
    run_train_if_needed(tlrs.train, 'models/fileLRS')

def run_train_xb():
    print("Checking and training XGBoost model if needed...")
    run_train_if_needed(txb.train, 'models/fileXB')

def run_train_rf():
    print("Checking and training Random Forest model if needed...")
    run_train_if_needed(trf.train, 'models/fileRF')

if __name__ == "__main__":
    run_train_lr()
    run_train_cnn()
    run_train_lrs()
    run_train_xb()
    run_train_rf()