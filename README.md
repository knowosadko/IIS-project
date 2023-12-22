# IIS-project - A Virtual Bartender Experience with Furhat
Project for the Intelligent Interactive Systems (IIS)
A virtual robot that adapts its behavior based on the emotions of users.

```bash
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
|   ├── SVC1.joblib
|   └── SVC2.joblib 
└── Scripts
    ├── stateDetection.py
    ├── face_detection.py
    ├── testnoface.py
    └── crop_and_sort.py
```

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

### SVC1.joblib
Support Vector Classification model ('coef0': 1, 'degree': 2, 'gamma': 0.5, 'kernel': 'poly') Accuracy: 0.640625

### model2.joblib
Random Forrest model {'bootstrap': False, 'criterion': 'log_loss', 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300} Accuracy: 0.6580645161290323

## Scripts
Directory for storing script files for the system

### crop_and_sort.py
Script for cropping out the faces in MultiEmoVA dataset and for sorting them acording to emotion.

### face_detection.py
Script for detecting faces and predicting their emotion real time.

### stateDetection
Script for training Machine Learning models.


# Resources
Microsoft Azure Speech services, it works better than google, has lower WER (word error rate).

Configuration in Web interface->Settings->Recognizer:

Region: North-europe

Key: ddb3fa21571e4e71adef5cfbafb9b4ee