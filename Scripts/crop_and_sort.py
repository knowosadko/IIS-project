import os
from feat import Detector
import imageio.v3 as iio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings


detector = Detector(device="cuda")

emotion_list = ["anger","disgust","fear","happiness","sadness","surprise","neutral"]



def proc_image(categories,img_name):
    path = os.path.join(os.getcwd(),"Data","multiEmoCrop")
    fp = os.path.join(path,"img_to_treat",categories, img_name)
    print(fp)
    image = iio.imread(fp)

    prediction = detector.detect_image(fp)
    aus = prediction.aus
    if aus.dropna().shape[0] != 0:

        face_info = prediction.facebox
        humor_info = prediction.emotions
        # print(humor_info)


        for i in range(len(humor_info["neutral"])):
            if face_info["FaceRectX"][i] > 0 and face_info["FaceRectY"][i] > 0: # idk why, but sometimes i have negatives values
                point1 = (int(face_info["FaceRectX"][i]),int(face_info["FaceRectY"][i]))
                point2 = (int(face_info["FaceRectX"][i])+int(face_info["FaceRectWidth"][i]),int(face_info["FaceRectY"][i])+int(face_info["FaceRectHeight"][i]))
                # print(point1)
                # print(point2)
                crop = image[point1[1]:point2[1],point1[0]:point2[0]]
                # print(crop.shape)
                

                prediction_emotion_value = []

                for em in emotion_list:
                        prediction_emotion_value.append(humor_info[em][i])
                ind_emotion = pd.Series(prediction_emotion_value).idxmax()
                real_emotion = ""
                match ind_emotion:
                        case 0:
                            real_emotion = "angry"
                        case 1:
                            real_emotion = "disgust"
                        case 2:
                            real_emotion = "fear"
                        case 3:
                            real_emotion = "happy"
                        case 4:
                            real_emotion = "sad"
                        case 5:
                            real_emotion = "surprise"
                        case 6:
                            real_emotion = "neutral"
                # print(real_emotion)
                # print(prediction_emotion_value[ind_emotion])
                if prediction_emotion_value[ind_emotion] > 0.9: #save image only if the emotion score is above 0.9
                    img_save = str(i) + "_" + img_name
                    # print(os.path.join(path,real_emotion,img_save))
                    iio.imwrite(os.path.join(path,real_emotion,img_save),crop)




def main():

    path = os.path.join(os.getcwd(),"Data","multiEmoCrop")
    img_categories = os.listdir(os.path.join(path,"img_to_treat"))
    print(img_categories)
    for cat in img_categories:
        img_list = os.listdir(os.path.join(path,"img_to_treat",cat))
        for img in img_list:
            proc_image(cat,img)
            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()