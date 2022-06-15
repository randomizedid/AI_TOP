#Hi! This code is meant to provide a demo of the Action Recognition model we are building for AI TOP. The model uses 2 stacked machine learning models, one is an open-source tool called Mediapipe that allows for real-time human bodyparts tracking, and on top of that there's an LSTM model that recognizes variable length sequences of actions and categorizes them. Right now, what it can detect is: no action, hand_biting, head scratching, covering face and covering ears, but it will be upgraded. Apart from the demo, there is also the chance to collect data, by checking the authorization box on the GUI. Once data are collected, you can send them to me to francesco.bonacini2021@my.ntu.ac.uk so that they can get labelled and added to the model. Thanks!

#Importing needed libraries
from lib2to3.pgen2.token import LEFTSHIFT
from msilib.schema import CheckBox
from tkinter import *
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from functions import *
import os
import random
import string

#Let's define a few functions that will be needed in the code.
def mediapipe_detection(image, model):
    #This function takes in an image and a mediapipe model and outputs the image and the prediction results.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results, drawing_spec_circle, drawing_spec_line):
    #This function takes in the image and the array of predictions and draws the predicted points on the image, followind the specs for radius and connections.
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_circle, connection_drawing_spec=drawing_spec_line)

def extract_keypoints(results):
    #This function takes in the prediction results and returns the array of landmarks.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

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

#Main starting!!!

#Initializing Keras model: it has 3 layers of LSTM and 3 Dense layers, every layer has a 0.2 dropout. The model uses ragged tensors to handle the variable sequence length. One thing that is missing is the data augmentation, since ragged tensors doesn't seem too easy to handle, but I'll be working on it.
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[None, 1662], ragged=True),
    tf.keras.layers.LSTM(64, activation = 'tanh', dropout=0.2, return_sequences = True),
    tf.keras.layers.LSTM(128, activation = 'tanh', dropout=0.2, return_sequences = True),
    tf.keras.layers.LSTM(64, activation = 'tanh', dropout=0.2),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation = 'softmax'),
])

#Preparing model weights and actions to recognize
model.load_weights(os.path.join(os.getcwd(), "model_weights\\5_actions_0.844.h5"))
actions = np.array(['hand_bite', 'cover_face', 'head_scratch', 'cover_ears', 'no_action'])

#Initializing GUI
root = Tk()
root.title("AI TOP Demo")
#root.geometry("800x700")

root.grid_rowconfigure(0, weight = 1)
root.grid_columnconfigure(0, weight = 1)

#Setting up the frame that will contain the predict button.
prediction_frame = LabelFrame(root)
prediction_frame.pack(fill = 'x', side = BOTTOM)

#Setting up the texts
label1 = Label(prediction_frame, text="Hello and welcome to the Action Recognition Demo! Up to now, the model can only detect hand biting, head scratching, covering face and covering ears, other actions should get labelled as 'no action'. \nNote that, by checking the box, your action execution will be recorded to be subsequently labelled and used to improve the model.", font= ("Arial", 8))
label2 = Label(prediction_frame, text="If you wish to help and authorize us to collect your data, please check the box. Thanks!", font= ("Arial", 8))
label1.pack()
label2.pack()

#Setting up the checkbox.
checkbox = ttk.Checkbutton (prediction_frame, text = "Yes, I give my permission to collect my data.", command = toggle)
checkbox.pack()

#Setting up the predict button.
predict_butt = Button(prediction_frame, text = "Start the Demo!", command = predict, width = 15)
predict_butt.pack()

#Running the Tkinter loop that handles and updates the GUI.
root.mainloop()