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

    counter = 0

    for categorie in categories:
        img_path = os.path.join(data_path, categorie)
        images = os.listdir(img_path)
        for file in images:
            if counter == 5: # Remove when testing done
                break
            file_path = os.path.join(img_path,file)

            detector = Detector(device="cuda")
            aus_data = detector.detect_image(file_path)

            if len(data) == 0:
                columnNames = aus_data.au_columns
            labels.append(categorie)
            data.append(aus_data.loc[0].aus.values.flatten().tolist())

            counter += 1

        print(columnNames)
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

    #Training
    labels, data = loadTrainingData(os.path.join(path,"Data","DiffusionFER","DiffusionEmotion_S","cropped"))
    


if __name__ == "__main__":
    main()