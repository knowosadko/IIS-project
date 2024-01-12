# IIS-project - A Virtual Bartender Experience with Furhat
Project for the Intelligent Interactive Systems (IIS)
A virtual robot that adapts its behavior based on the emotions of users.

To run place the working directory in the ISS-PROJECT folder and run Scripts/main.py

```bash
├──requirements.txt
├── Data
|   ├── trainAUs.csv
|   ├── trainLabels.csv
|   ├── DiffusionFER
|   |   └── DiffusionEmotion_S
|   |       └── cropped
|   |           ├── angry
|   |           ├── disgust
|   |           ├── fear
|   |           ├── happy
|   |           ├── neutral
|   |           ├── sad
|   |           └── surprise
|   ├── multiEmoCrop
|   |           ├── angry
|   |           ├── disgust
|   |           ├── fear
|   |           ├── happy
|   |           ├── img_to_treat
|   |           ├── neutral
|   |           ├── sad
|   |           └── surprise
|   └── mixed_DataSET
|               ├── angry
|               ├── disgust
|               ├── fear
|               ├── happy
|               ├── neutral
|               ├── sad
|               └── surprise
├── Models
|   ├── FER_2013_RF.joblib
|   ├── model2.joblib
|   ├── SVC1.joblib
|   └── SVC2.joblib 
└── Scripts
    ├── main.py
    ├── crop_and_sort.py
    ├── face_detection.py
    ├── ISS.py
    ├── globals.py
    ├── stateDetection.py
    ├── testnoface.py
    └── texts.py
    
```

## requirements.txt
List of versions. Generated with "pip3 freeze > requirements.txt".

## Data
Directory for storing data for training the models

### DiffusionFER
Directory containing the DiffusionFER dataset (not included in the git because of size, source https://huggingface.co/datasets/FER-Universe/DiffusionFER)

### multiEmoCrop
Directory containing the MultiEmoVA dataset (not included in the git because of its size)

### trainAUs.csv
CSV file storing Action Units detected in images in DiffusionFER

### trainLabels.csv
CSV file storing Labels for the trainAUs.csv

## Models
Dictionary for storing trained models

### FER_2013_RandomForest.joblib
Random Forrest model ('bootstrap': True, 'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200)
Trained on 20096 faces from the FER_2013 dataset
Evaluated on 2870 faces
Test Accuracy: 0.6153846153846154
Test Time: 0.00996708869934082

### model2.joblib
Random Forrest model ('bootstrap': False, 'criterion': 'log_loss', 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300)
Test Accuracy: 0.6580645161290323
Test Time: 0.015626192092895508

### SVC1.joblib
Support Vector Classification model ('coef0': 1, 'degree': 2, 'gamma': 0.5, 'kernel': 'poly') 
Test Accuracy: 0.640625
Test Time: 0.0032041072845458984

### SVC2.joblib
Support Vector Classification model ('coef0': 1, 'degree': 2, 'gamma': 1, 'kernel': 'poly')
Test Accuracy: 0.6440
Test Time: 0.005008697509765625

### model2.joblib
Random Forrest model {'bootstrap': False, 'criterion': 'log_loss', 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300} 
Accuracy: 0.6580645161290323

## Scripts
Directory for storing script files for the system

### main.py
Main script for running the face detection in parralell

### crop_and_sort.py
Script for cropping out the faces in MultiEmoVA dataset and for sorting them acording to emotion.

### face_detection.py
Old script for detecting faces and predicting their emotion real time.

### stateDetection
Script for training Machine Learning models.

# Resources
Microsoft Azure Speech services, it works better than google, has lower WER (word error rate).

Configuration in Web interface->Settings->Recognizer:

Region: North-europe

Key: ddb3fa21571e4e71adef5cfbafb9b4ee
