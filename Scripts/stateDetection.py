import os
import numpy as np
import pandas as pd
from joblib import dump, load
from feat import Detector

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

def train_models(data):
    # Function for training modules on train data
    ...

def use_model(data, modelPath):
    # Function for using premade model to classify AUS data
    module = load(modelPath)
    predictions = module.predict(data)

def main():
    path = os. getcwd()

    #Training Data
    labels, data = loadTrainingData(os.path.join(path,"Data","DiffusionFER","DiffusionEmotion_S","cropped"))
    data.to_csv(os.path.join(path,"Data","trainAUs.csv"),index=False)
    labels.to_csv(os.path.join(path,"Data","trainLabels.csv"),index=False)
    


if __name__ == "__main__":
    main()