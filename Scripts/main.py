import os
from time import sleep
import threading
from feat.detector import Detector
import cv2
import pandas as pd
import warnings
from ISS import main_tree
import globals
#import ISS

def faceDetection():
    # Function for running face detection

    warnings.filterwarnings("ignore") # Turn of warnings for thread

    from stateDetection import load_model # import state detection to thread

    # Set up path
    path = os. getcwd() # Make sure running from main folder

    # Set up detector
    detector = Detector(device="cuda") 

    #emotion = None # Initiates the emotion variable
    modelName = "FER_2013_RandomForest.joblib" #Change to change model
    model = load_model(modelName)

    # Start vidio capture
    cap = cv2.VideoCapture(0)

    # Set up colums for storing data
    columns = ['Frame'] + [f'AU{i}' for i in ['01', '02', '04', '05', '06', '07', '09', '10', '11', '12', '14', '15', '17', '20', '23', '24', '25', 26, 28, 43]]
    aus_data = pd.DataFrame(columns=columns)

    try:
        frame_number = 0
        frame_skip = 20  # Adjust this value to change how often landmarks and AUs are detected, higher value == less stutters downside = Less AU info
        current_frame = 0
        emotion = None
        while True: # detect faces
            ret, frame = cap.read()
            faces = detector.detect_faces(frame)
            
            try:
                face = faces[0][0] # Select "first" face
                detecting_face = True
                (x0, y0, x1, y1, p) = face

                # Display face detection
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)


                if current_frame % frame_skip == 0: # Frame not skipped
                    # Predic emotion
                    landmarks = detector.detect_landmarks(frame, faces)
                    aus = detector.detect_aus(frame, landmarks)[0][0]
                    globals.semaphor.acquire()             
                    globals.emotion = model.predict(pd.DataFrame([list(aus)], columns=columns[1:]))
                    globals.semaphor.release()
                    # Storing data (disabled)
                    frame_data = pd.DataFrame([[frame_number] + list(aus)], columns=columns)
                    aus_data = pd.concat([aus_data, frame_data], ignore_index=True)
                    #aus_data.to_csv('data.csv', index=False)

                globals.semaphor.acquire()             
                emotion = globals.emotion
                globals.semaphor.release()
                # Display emotion
                if globals.emotion != None:
                    cv2.putText(frame, emotion[0], (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            except IndexError as e:
                #print(f"No face detected")
                detecting_face = False
                emotion = None

            cv2.imshow('Face Detection', frame) # Show camera

            if cv2.waitKey(1) == 27: #press escape to quit
                break

            # Increase counter
            frame_number += 1
            current_frame += 1

            if current_frame % frame_skip == 0: # Make some time for the ISS to run
                sleep(0.01)
    except KeyboardInterrupt:
        pass

    finally: # Turning off
        cap.release()
        cv2.destroyAllWindows()

    #aus_data.to_csv('data.csv', index=False)


if __name__ == "__main__":

    # Setting up variable for storing emotional state

    # Make thread for face detection
    thread_stateDetection = threading.Thread(target=faceDetection,args=())
    
    # Make thread for main tree for furhat
    thread_ISS = threading.Thread(target=main_tree)
    thread_stateDetection.start()
    thread_ISS.start()


