import os
import numpy as np
import pandas as pd
from joblib import dump, load
from feat import Detector

def loadTrainingData(data_path):
    # Function for loading a dictinary of images as inpuit data (Used for training)
    categories = os.listdir(data_path)

    labels = []
    aus_data = []

    counter = 0

    for categorie in categories:
        img_path = os.path.join(data_path, categorie)
        images = os.listdir(img_path)
        for file in images:
            if counter == 1: # Remove when testing done
                break
            file_path = os.path.join(img_path,file)
            # replace with detector

            detector = Detector(device="cuda")
            data = detector.detect_image(file_path)

            labels.append(categorie)
            aus_data.append(data.aus)

            counter = 1

    return labels, aus_data

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
    print(data)

    #aus = pd.read_csv(os.path.join(path,"processed","aus.csv")) # Should be real time for final version

if __name__ == "__main__":
    main()