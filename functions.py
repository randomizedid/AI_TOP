import os
import numpy as np
import cv2
import mediapipe as mp
import random

#setting up the mediapipe model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#defining a few functions to detect, draw and extract keypoints
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results, drawing_spec_circle, drawing_spec_line):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def collect_data(action):
    #recording data

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #drawing specs for face dots
    drawing_spec_circle = mp_drawing.DrawingSpec()
    drawing_spec_circle.circle_radius = 1
    drawing_spec_circle.thickness = 1
    drawing_spec_circle.color = (0,0,255)

    #drawing specs for face connections
    drawing_spec_line = mp_drawing.DrawingSpec()
    drawing_spec_line.thickness = 1

    #check how many sequences' folder exist to not overwrite anything, this way one can record in different sessions
    try:
        os.makedirs(os.path.join(os.getcwd(), "DATA", action))
    except:
        print("not the first recording session for this action, I guess")

    folder_list = []
    for folder in os.listdir(os.path.join(os.getcwd(), "DATA", action)):
        folder_list.append(int(folder))
    if not folder_list:
        number_of_sequences = 0
    else:
        number_of_sequences = max(folder_list) + 1

    while True:
        with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
            
            sequence_length = random.randint(10,40)
            #create new sequence folder
            try:
                os.makedirs(os.path.join(os.getcwd(), "DATA", action, str(number_of_sequences)))
            except:
                print("folder already existing, something's wrong with the code...")

            for frame_num in range(sequence_length):

                #read camera
                ret,frame = cap.read()

                #make detections
                image, results = mediapipe_detection(frame, holistic)
                #print(results.multi_face_landmarks[0].landmark[1].x)

                #draw landmarks
                draw_landmarks(image, results, drawing_spec_circle, drawing_spec_line)
                
                #collection logic
                if frame_num == 0:
                    cv2.putText(image, 'Action: {} is being collected'.format(action), (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, number_of_sequences), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('feed', image)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, number_of_sequences), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('feed', image)
                    
                keypoints = extract_keypoints(results)

                #save data
                npy_path = os.path.join(os.getcwd(), "DATA", action, str(number_of_sequences), str(frame_num))
                np.save(npy_path, keypoints)

                #close loop
                if cv2.waitKey(10) & 0XFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
            
            number_of_sequences+=1
    return






