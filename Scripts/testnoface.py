import os
import numpy as np
import pandas as pd
from feat import Detector

detector = Detector(device="cuda")
path = os.path.join(os.getcwd(),"Data","NoFaceTest")

data =[]
for file in os.listdir(path):
    img = os.path.join(path, file)
    aus_data = detector.detect_image(img)

    if (aus_data.dropna().shape[0] != 0):
        data.append(aus_data.loc[0].aus.values.flatten().tolist())

dataset = pd.DataFrame(np.array(data))

print(dataset)