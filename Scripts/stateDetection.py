import os
import numpy as np
import pandas as pd
from joblib import dump, load

def loadDictionaryData(img_path)
    # Function for loading a dictinary of images as inpuit data (Used for training)
     images = os.listdir(img_path)

def train_models(data)
    # Function for training modules on train data

def use_model(data, modelPath):
    # Function for using premade model to classify AUS data
    module = load(modelPath)
    predictions = module.predict(data)

def main():
    path = os. getcwd()

    #Training
    img_path = os.path.join(path,"Data")

    #aus = pd.read_csv(os.path.join(path,"processed","aus.csv")) # Should be real time for final version

if __name__ == "__main__":
    main()