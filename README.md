# IIS-project - A Virtual Bartender Experience with Furhat
Project for the Intelligent Interactive Systems (IIS)
A virtual robot that adapts its behavior based on the emotions of users.

IIS-project
├── Data
|   ├── trainAUs.csv
|   ├── trainLabels.csv
|   └── DiffusionFER
|       └── DiffusionEmotion_S
|           └── cropped
|               ├── angry
|               ├── disgust
|               ├── fear
|               ├── happy
|               ├── sad
|               └── surprise
└── Scripts
    └──stateDetection.py

## Scripts
Directory for storing script files for the system

## Data
Directory for storing data for training the models

### DiffusionFER
Directory containing the DiffusionFER dataset (not included in the git because of size, source https://huggingface.co/datasets/FER-Universe/DiffusionFER)

### trainAUs.csv
CSV file storing Action Units detected in images in DiffusionFER

### trainLabels.csv
CSV file storing Labels for the trainAUs.csv

## Models
Dictionary for storing trained models