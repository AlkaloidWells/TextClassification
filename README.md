# Text Classification Program

This program is designed to classify text queries into predefined categories using various machine learning models. The models include Logistic Regression, Logistic Regression with SMOTE, CNN, XGBoost, and Random Forest. This README file provides instructions on how to train and test the classifiers.

## Prerequisites

Before running the program, ensure you have the following installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```

.
├── Classifiers
│   ├── textclassiferLR.py
│   ├── textclassiferLRS.py
│   ├── textclassiferCNN.py
│   ├── textclassiferXB.py
│   ├── textclassiferRF.py
├── Trainers
│   ├── trainLR.py
│   ├── trainLRS.py
│   ├── trainCNN.py
│   ├── trainXB.py
│   ├── trainRF.py
├── helper
│   ├── utils.py
├── models
│   ├── fileLR
│   ├── fileLRS
│   ├── fileCNN
│   ├── fileXB
│   ├── fileRF
├── testTR.py
├── testCF.py
├── requirements.txt
└── README.md



## Training the Models
To train the models, run the testTR.py script. This script checks if the model directories exist and are not empty. If a directory does not exist or is empty, it creates the directory and trains the corresponding model.

'''python testCF.py'''


The script will print messages indicating the progress of the training process for each model.

Testing the Classifiers
To test the classifiers, run the testCF.py script. This script uses the trained models to classify a sample query and prints the results for each model.

python testCF.py

The script will print the classification results for the sample query using each of the trained models.

Example Usage
Training the Models


### Output:

Checking and training Logistic Regression model if needed...
Training model and saving to models/fileLR...
...
Checking and training CNN model if needed...
Training model and saving to models/fileCNN...
...
Checking and training Logistic Regression with SMOTE model if needed...
Training model and saving to models/fileLRS...
...
Checking and training XGBoost model if needed...
Training model and saving to models/fileXB...
...
Checking and training Random Forest model if needed...
Training model and saving to models/fileRF...
...


 ## Testing the Classifiers

### Output:
Model 1 (Logistic Regression without SMOTE) results:
   category_id  probability category_name
0            1     0.75     Electronics
1            2     0.25     Mobile Phones

Model 2 (Logistic Regression with SMOTE) results:
   category_id  probability category_name
0            1     0.80     Electronics
1            2     0.20     Mobile Phones

Model 3 (CNN) results:
   category_id  probability category_name
0            1     0.85     Electronics
1            2     0.15     Mobile Phones

Model 4 (XGBoost) results:
   category_id  probability category_name
0            1     0.90     Electronics
1            2     0.10     Mobile Phones

Model 5 (Random Forest) results:
   category_id  probability category_name
0            1     0.88     Electronics
1            2     0.12     Mobile Phones




## Notes

Ensure that the models directory and its subdirectories have the correct permissions for reading and writing files.
Modify the sample query in testCF.py to test different queries.
License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Thanks to the contributors and the open-source community for their support and contributions.
