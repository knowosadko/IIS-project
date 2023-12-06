import os
import numpy as np
import pandas as pd
from joblib import dump, load
from feat import Detector
from sklearn.model_selection import train_test_split

def loadTrainingData(data_path):
    # Function for loading a dictinary of images as inpuit data (Used for training)
    categories = os.listdir(data_path)

    labels = []
    data = []

    detector = Detector(device="cuda")

    for categorie in categories:
        img_path = os.path.join(data_path, categorie)
        images = os.listdir(img_path)

        for file in images:
            file_path = os.path.join(img_path,file)

            aus_data = detector.detect_image(file_path)

            if (aus_data.dropna().shape[0] != 0): # Only store detected faces
                if len(data) == 0: # Get columns only once
                    columnNames = aus_data.au_columns
                
                labels.append(categorie)
                data.append(aus_data.loc[0].aus.values.flatten().tolist())

        labelset = pd.DataFrame(np.array(labels), columns=["emotion"])

        # convert to pandas format
        dataset = pd.DataFrame(np.array(data), columns=columnNames)

    return labelset, dataset

def loadTrainingCSV(data_path,label_path):
    dataset = pd.read_csv(data_path)
    labelset = pd.read_csv(label_path)

    return labelset, dataset

def splitTrainValTest(data, labels, sizeTest, sizeVal):
    # Function for spliting data into training, validation and test sets
    data_in, test_in, data_out, test_out = train_test_split(data, labels, test_size=sizeTest, stratify=labels)
    train_in, val_in, train_out, val_out = train_test_split(data_in, data_out, test_size=(sizeVal/(1-sizeTest)), stratify=data_out)
    return train_in, train_out, val_in, val_out, test_in, test_out 

def train_models(train_data, train_labels, val_data, val_labels):
    ...

def use_model(data, modelPath):
    # Function for using trained model to classify AUS data
    module = load(modelPath)
    predictions = module.predict(data)

def main():
    path = os. getcwd()

    #Training Data
    #labels, data = loadTrainingData(os.path.join(path,"Data","DiffusionFER","DiffusionEmotion_S","cropped"))
    #data.to_csv(os.path.join(path,"Data","trainAUs.csv"),index=False)
    #labels.to_csv(os.path.join(path,"Data","trainLabels.csv"),index=False)
    labels, data = loadTrainingCSV(os.path.join(path,"Data","trainAUs.csv"), os.path.join(path,"Data","trainLabels.csv"))
    
    train, train_labels, val, val_labels, test, test_labels = splitTrainValTest(data, labels, 0.1, 0.2)
    print(train)

if __name__ == "__main__":
    main()