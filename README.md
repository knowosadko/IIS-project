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
|   └── multiEmoCrop
|               ├── angry
|               ├── disgust
|               ├── fear
|               ├── happy
|               ├── img_to_treat
|               ├── neutral
|               ├── sad
|               └── surprise
├── Models
|   └── SVC1.joblib
└── Scripts
    └── stateDetection.py
```

## Data
Directory for storing data for training the models

### DiffusionFER
Directory containing the DiffusionFER dataset (not included in the git because of size, source https://huggingface.co/datasets/FER-Universe/DiffusionFER)

### multiEmoCrop

### trainAUs.csv
CSV file storing Action Units detected in images in DiffusionFER

### trainLabels.csv
CSV file storing Labels for the trainAUs.csv

## Models
Dictionary for storing trained models

### SVC1.joblib
Support Vector Classification model ('coef0': 1, 'degree': 2, 'gamma': 0.5, 'kernel': 'poly') Accuracy: 0.640625

## Scripts
Directory for storing script files for the system

### stateDetection
Script for training models