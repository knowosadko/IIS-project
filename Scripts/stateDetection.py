import os
import time
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load as load_
from feat import Detector
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import warnings


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

def train_models(model,train_data, train_labels, val_data, val_labels, param_grid=[]):

    if (len(param_grid) > 0):
        model = GridSearchCV(model, param_grid)

    model.fit(train_data, train_labels)

    if (len(param_grid) > 0):
        tic = time.time()
        predicted_val = model.best_estimator_.predict(val_data)
    else:
        tic = time.time()
        predicted_val = model.predict(val_data)
    toc = time.time()
    process_time = toc - tic
    accuracy = accuracy_score(predicted_val, val_labels)

    return model, accuracy, process_time

def store_model(model, modelName):
    modelPath = os.path.join(os.getcwd(), "Models", modelName)
    dump(model, model_path)

def load_model(modelName):
    modelPath = os.path.join(os.getcwd(), "Models", modelName)
    return load_(modelPath)

def evaluate_model(model, data, labels):
    # Function for evalating trained model on test data

    tic = time.time()
    predictions = model.predict(data)
    toc = time.time()
    process_time = toc - tic
    accuracy = accuracy_score(predictions, labels)
    return predictions, accuracy, process_time

def main():
    path = os.getcwd()

    #Training Data
    #labels, data = loadTrainingData(os.path.join(path,"Data","DiffusionFER","DiffusionEmotion_S","cropped"))
    #data.to_csv(os.path.join(path,"Data","trainAUs.csv"),index=False)
    #labels.to_csv(os.path.join(path,"Data","trainLabels.csv"),index=False)
    labels, data = loadTrainingCSV(os.path.join(path,"Data","trainAUs.csv"), os.path.join(path,"Data","trainLabels.csv"))
    
    train, train_labels, val, val_labels, test, test_labels = splitTrainValTest(data, labels, 0.1, 0.2)

    model1, model1_accuracy, model1_time = train_models(SVC(),train,train_labels,val,val_labels)
    print(f"Model: SVC \n Accuracy: {model1_accuracy} \n Time: {model1_time} s\n")

    model2 = sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    model2, model2_accuracy, model2_time = train_models(model2,train,train_labels,val,val_labels)
    print(f"Model: SG \n Accuracy: {model2_accuracy} \n Time: {model2_time} s\n")

    model3 = KNeighborsClassifier()
    model3, model3_accuracy, model3_time = train_models(model3,train,train_labels,val,val_labels)
    print(f"Model: Nearest neighbors \n Accuracy: {model3_accuracy} \n Time: {model3_time} s\n")

    #Model: SVC 
    #Accuracy: 0.62890625
    #Time: 0.012627601623535156 s

    #Model: SG
    #Accuracy: 0.6015625
    #Time: 0.0009980201721191406 s

    #Model: Nearest neighbors 
    #Accuracy: 0.57421875
    #Time: 0.11684870719909668 s

    # SVC best performance and decent time

    # Vector tuning Support Vector Classification
    param_grid = [
        {"kernel": ["linear"]},
        {"kernel": ["poly"], "degree": [1,2,3, 5, 10], "gamma": ["scale","auto",0,0.5,1,5,10,20], "coef0": [0,1]},
        {"kernel": ["rbf"], "gamma": ["scale","auto",0,0.5,1,5,10,20]},
        {"kernel": ["sigmoid"], "gamma": ["scale","auto",0,0.5,1,5,10,20], "coef0": [0,1]}
    ]
    best_model, best_model_accuracy, best_model_time = train_models(SVC(),train,train_labels,val,val_labels,param_grid)
    print(f"Best parameters of SVC: {best_model.best_params_}\n Accuracy: {best_model_accuracy} \n Time: {best_model_time} s\n")
    #Accuracy: 0.625
    #Time: 0.003987789154052734 s

    # Save model
    modelName = "SVC2.joblib"
    store_model(best_model.best_estimator_, modelName)

    # Validate on testset
    model = load_model(modelName)
    presictions, accuracy, time_ = evaluate_model(model, test, test_labels)
    print(f"On test set: \n\t Accuracy: {accuracy}\n\t Time: {time_}")

    # SCV1.joblib
    #Parameters: {'coef0': 1, 'degree': 2, 'gamma': 0.5, 'kernel': 'poly'}
    #Accuracy: 0.640625
    #Time: 0.0032041072845458984

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()