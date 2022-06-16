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

def visualize_probabilities(res, actions, input_frame, colors):
    #This function takes in the results from the LSTM prediction and the image and draws a rectangle for every action to detect. It then fills every rectangle with a color, where the degree to which every rectangle gets filled is directly proportional to the probability detected for that action.
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def predict():
    #This function makes all the LSTM prediction.

    #Setting up specifics for the drawings.
    drawing_spec_circle = mp_drawing.DrawingSpec()
    drawing_spec_circle.circle_radius = 1
    drawing_spec_circle.thickness = 1
    drawing_spec_circle.color = (0,0,255)
    drawing_spec_line = mp_drawing.DrawingSpec()
    drawing_spec_line.thickness = 1

    #Other variables
    sequence = []
    predictions = [0]
    #threshold = 0.95
    colors = [(0,0,0), (155,155,0), (0,255,0), (255, 0, 0), (0, 0, 255)]

    #Choosing which camera to use, 0 is default webcam, if you have any other device just go up with the number.
    cap = cv2.VideoCapture(0)

    #Creates a folder in which to store the collected data, if it isn't already there.
    try:
        os.makedirs(os.path.join(os.getcwd(), "data_to_label"))
    except:
        pass
    
    #Checking the state of the checkbox, if it is checked, a video file will be initialized with a random name (hopefully with 52^14 possible combinations there will be no repeated names, otherwise the older video will be overwritten).
    a = checkbox.state()
    if len(a) > 1:
        char_to_choose = string.ascii_letters
        name = ''.join(random.choice(char_to_choose) for i in range(14))
        #video saving parameters
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("data_to_label\\" + name + ".mp4", fourcc, 6.0, (640, 480))

    #Initializes the mediapipe model for the real-time detection, opens up the camera and starts getting frames. If the box is checked, frames will be saved to the previously initialized video file.
    with mp_holistic.Holistic(min_detection_confidence=0.8) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            if len(a) > 1:
                out.write(frame)

            #Takes in the frame and draws the mediapipe landmarks
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results, drawing_spec_circle, drawing_spec_line)

            #Extracts keypoints and creates sequences that will be used for LSTM prediction. The problem here is that while in training it is fairly easy to use variable length sequences with ragged tensors, in prediction is it much harder, because you are predicting actions in real-time and you don't know how long that action will last. Therefore, one has to choose a fixed length and predict sequences of that length (have to say, it works pretty decently). We need to find a solution to this, if possible (an idea might be to predict at the same time on different sequences, like at a given time you have 10 sequences of length 5-10-15 etc. frames and you run the prediction on all of them, returning the highest probability among them. Might be computationally undoable.) 
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-8:]
            
            #When the sequence hits the right number of frames, the model performs the prediction on it, prints out what it thinks the action is, and fills the rectangles with probabilities.
            if len(sequence) == 8:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                image = visualize_probabilities(res, actions, image, colors)

            #This control makes the window resizable.
            cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('cai', 1300, 900)
            #Showing the processed image (it has both landmarks drawn and probabilities' rectangles filled with colours).
            cv2.imshow('Demo', image)

            #This control is used to end the prediction loop. Clicking on the 'x' of the window won't help, only 'q' works.
            if cv2.waitKey(10) & 0XFF == ord('q'):
                break
        
        #Releases the camera and closes the prediction window when the loop is interrupted.
        cap.release()
        cv2.destroyAllWindows()
    return

def toggle():
    #This function prints the state of the checkbox, it is actually not used anywhere in the code as I am writing this.
    a = checkbox.state()
    print(a)
    if a[2] == 'selected':
        print("open")





