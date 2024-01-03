import os
import time
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load as load_
from feat import Detector
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
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
    dump(model, modelPath)

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
    # labels, data = loadTrainingData(os.path.join(path,"Data","FER_2013"))
    # data.to_csv(os.path.join(path,"Data","FER_2013_trainAUs.csv"),index=False)
    # labels.to_csv(os.path.join(path,"Data","FER_2013_trainLabels.csv"),index=False)


    labels, data = loadTrainingCSV(os.path.join(path,"Data","trainAUs.csv"), os.path.join(path,"Data","trainLabels.csv"))
    
    train, train_labels, val, val_labels, test, test_labels = splitTrainValTest(data, labels, 0.1, 0.2)

    main1(train, train_labels, val, val_labels, test, test_labels)
    # main2(train, train_labels, val, val_labels, test, test_labels)
    # # main3(train, train_labels, val, val_labels, test, test_labels)

def main1(train, train_labels, val, val_labels, test, test_labels):
    model1, model1_accuracy, model1_time = train_models(SVC(),train,train_labels,val,val_labels)
    print(f"Model: SVC \n Accuracy: {model1_accuracy} \n Time: {model1_time} s\n")

    model2 = sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    model2, model2_accuracy, model2_time = train_models(model2,train,train_labels,val,val_labels)
    print(f"Model: SG \n Accuracy: {model2_accuracy} \n Time: {model2_time} s\n")

    model3, model3_accuracy, model3_time = train_models(KNeighborsClassifier(),train,train_labels,val,val_labels)
    print(f"Model: Nearest neighbors \n Accuracy: {model3_accuracy} \n Time: {model3_time} s\n")

    model4, model4_accuracy, model4_time = train_models(DecisionTreeClassifier(), train, train_labels, val, val_labels)
    print(f"Model: tree \n Accuracy: {model4_accuracy} \n Time: {model4_time} s\n")

    model5, model5_accuracy, model5_time = train_models(RandomForestClassifier(), train, train_labels, val, val_labels)
    print(f"Model: randomforest \n Accuracy: {model5_accuracy} \n Time: {model5_time} s\n")

    model6, model6_accuracy, model6_time = train_models(GaussianProcessClassifier(), train, train_labels, val, val_labels)
    print(f"Model: gausian \n Accuracy: {model6_accuracy} \n Time: {model6_time} s\n")

    #SVC Accuracy: 0.6015625 Time: 0.013618230819702148 s
    #SG  Accuracy: 0.5859375 Time: 0.0019192695617675781 s
    #Nearest neighbors  Accuracy: 0.5546875 Time: 0.12466597557067871 s
    #tree Accuracy: 0.484375 Time: 0.0010335445404052734 s
    #randomforest  Accuracy: 0.62109375 Time: 0.0064849853515625 s
    #gausian Accuracy: 0.59765625 Time: 0.28975892066955566 s

    
def main2(train, train_labels, val, val_labels, test, test_labels):
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
    modelName = "FER_2013_SVC2.joblib"
    store_model(best_model.best_estimator_, modelName)

    # Validate on testset
    model = load_model(modelName)
    presictions, accuracy, time_ = evaluate_model(model, test, test_labels)
    print(f"On test set: \n\t Accuracy: {accuracy}\n\t Time: {time_}")


def main3(train, train_labels, val, val_labels, test, test_labels):
    param_grid = {"criterion":["gini", "entropy", "log_loss"], 
                  "n_estimators": [50, 100, 150, 200, 250, 300, 500],
                  "bootstrap": [True, False],
                  "max_features": ["sqrt","log2", None], 
                  "min_samples_split": [2, 5, 10],
                  "min_samples_leaf": [1, 2, 4]}
    
    best_model, best_model_accuracy, best_model_time = train_models(RandomForestClassifier(),train,train_labels,val,val_labels,param_grid)
    print(f"Best parameters of SVC: {best_model.best_params_}\n Accuracy: {best_model_accuracy} \n Time: {best_model_time} s\n")

    # Save model
    modelName = "model2.joblib"
    store_model(best_model.best_estimator_, modelName)

    # Validate on testset
    model = load_model(modelName)
    presictions, accuracy, time_ = evaluate_model(model, test, test_labels)
    print(f"On test set: \n\t Accuracy: {accuracy}\n\t Time: {time_}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()