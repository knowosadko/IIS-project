import os
from feat.detector import Detector
import cv2
import pandas as pd
import warnings

import stateDetection

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Set up path
    path = os. getcwd() # Make sure running from main folder
    
    # Set up detector
    detector = Detector(device="cpu") 

    emotion = None # Initiates the emotion variable
    modelName = "FER_2013_RF.joblib" #Change to change model
    model = stateDetection.load_model(modelName)

    # Start vidio capture
    cap = cv2.VideoCapture(0)
    
    # Set up colums for storing data
    columns = ['Frame'] + [f'AU{i}' for i in ['01', '02', '04', '05', '06', '07', '09', '10', '11', '12', '14', '15', '17', '20', '23', '24', '25', 26, 28, 43]]
    aus_data = pd.DataFrame(columns=columns)

    try:
        frame_number = 0
        frame_skip = 20  # Adjust this value to change how often landmarks and AUs are detected, higher value == less stutters downside = Less AU info
        current_frame = 0

        while True:
            ret, frame = cap.read()
            faces = detector.detect_faces(frame)

            try:
                face = faces[0][0]
                detecting_face = True
                (x0, y0, x1, y1, p) = face

                # Display face detection
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)


                if current_frame % frame_skip == 0:
                    # Predic emotion
                    landmarks = detector.detect_landmarks(frame, faces)
                    aus = detector.detect_aus(frame, landmarks)[0][0]
                                        
                    emotion = model.predict(pd.DataFrame([list(aus)], columns=columns[1:]))

                    # Storing data (disabled)
                    frame_data = pd.DataFrame([[frame_number] + list(aus)], columns=columns)
                    aus_data = pd.concat([aus_data, frame_data], ignore_index=True)
                    #aus_data.to_csv('data.csv', index=False)

                # Display emotion
                if emotion != None:
                    cv2.putText(frame, emotion[0], (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            except IndexError as e:
                #print(f"No face detected")
                detecting_face = False
                emotion = None

            #cv2.imshow('Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1
            current_frame += 1
    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

    #aus_data.to_csv('data.csv', index=False)
