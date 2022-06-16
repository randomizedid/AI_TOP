from lib2to3.pgen2.token import LEFTSHIFT
from msilib.schema import CheckBox
import sqlite3
from tkinter import *
from tkinter import ttk
from functools import partial
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from datetime import datetime

sys.path.insert(0, r"C:\Users\cocco\Desktop\collection_protocol")

from functions import *

#loop function for detection and prediction, I am building a GUI that has a button for prediction. it opens a cv2 window that can run at any fps
#defining a few functions to detect, draw and extract keypoints

def visualize_probabilities(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def predict():
    drawing_spec_circle = mp_drawing.DrawingSpec()
    drawing_spec_circle.circle_radius = 1
    drawing_spec_circle.thickness = 1
    drawing_spec_circle.color = (0,0,255)
    drawing_spec_line = mp_drawing.DrawingSpec()
    drawing_spec_line.thickness = 1
    sequence = []
    predictions = [0]
    threshold = 0.95
    colors = [(0,0,0), (155,155,0), (0,255,0), (255, 0, 0), (0, 0, 255)]

    cap = cv2.VideoCapture(0)
    now = random.randint(1,10000)
    #initializing variables
    try:
        os.makedirs(os.path.join(os.getcwd(), "data_to_label"))
    except:
        pass
    name = "data_to_label\\video_to_label" + str(now) + ".avi"
    #video saving parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name, fourcc, 6.0, (640, 480))

    with mp_holistic.Holistic(min_detection_confidence=0.8) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            a = checkbox.state()
            if len(a) > 1:
                
                #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                out.write(frame)

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results, drawing_spec_circle, drawing_spec_line)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-8:]
            
            if len(sequence) == 8:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                image = visualize_probabilities(res, actions, image, colors)

            cv2.imshow('feed', image)

            #close loop
            if cv2.waitKey(10) & 0XFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return

def toggle():
    a = checkbox.state()
    print(a)
    if a[2] == 'selected':
        print("open")

#Main starting!!!
#check_var = IntVar()

#initializing keras model
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
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation = 'softmax'),
])

model.load_weights(r"C:\Users\cocco\Desktop\collection_protocol\model_weights\\5_actions_844(to_test).h5")

#initializing variables
try:
    os.makedirs(os.path.join(os.getcwd(), "DATA"))
except:
    pass

actions = np.array(['hand_bite', 'cover_face', 'head_scratch', 'cover_ears', 'no_action'])

#initializing GUI
root = Tk()
root.title("Collection Protocol")
#root.geometry("800x700")

root.grid_rowconfigure(0, weight = 1)
root.grid_columnconfigure(0, weight = 1)

#predict frame
prediction_frame = LabelFrame(root, text = "Prediction")
prediction_frame.pack(fill = 'x', side = BOTTOM)

#text frame
frame2 = LabelFrame(root, text = "Protocol")
frame2.pack( side = LEFT)

#bullet point
txt = Text(frame2, wrap='word', height=32)
txt.pack()
txt.tag_configure('text_body', font=('Times', 14), lmargin1=0,
lmargin2=0, )
txt.tag_configure('bulleted_list', font=('Times', 14), lmargin1='10m',
lmargin2='15m', tabs=['15m'])
txt.tag_configure('text_body_important', font=('Times', 15), lmargin1=0,
lmargin2=0, underline=True)

txt.insert(END, u"This is the demo for the AI TOP project, that incorporates a data collection tool.\nThe demo starts by clicking on the Predict button on the bottom, while actions to recognize are listed on the right, make sure to have a webcam ready and not used by any other program.\n", 'text_body')
txt.insert(END, u"If you want to add more data to the dataset, choose an action and click on the relative button, the webcam will turn on, and a new window will appear showing a real time motion tracking rendering of what the webcam sees.\n", 'text_body')
txt.insert(END, u"Data collection will start immediately, iterating through this protocol:\n", 'text_body')
txt.insert(END, u"\u00B7\tA green writing appears for 1 second, saying 'Action: name_action is being collected'\n", 'bulleted_list')
txt.insert(END, u"\u00B7\tIt disappears, and waits for you to complete the action (the length of the action is variable and random, between 10 and 40 frames)\n", 'bulleted_list')
txt.insert(END, u"\u00B7\tOnce the random number of frames is collected, the green writing will appear again, giving you 1 second of pause\n", 'bulleted_list')
txt.insert(END, u"When you are finished collecting the actions (you can collect how many you want), just press Q and the collection will stop (it might be needed to press Q more than once if the system is performing a task).\n", 'text_body')
txt.insert(END, u"Please send me back the DATA folder that will be created.\n", 'text_body')
txt.insert(END, u"1) NOTE: no video of you will be recorded or saved, collected data are only coordinates of the points tracked by the algorithm.\n", 'text_body_important')
txt.insert(END, u"2) NOTE: Do not perform the movement that bring to the action, but only the action itself. Example: in the cover_face action we don't need the data of you raising your hands to your face, therefore just keep covering your face throughout the entire session of recording.", 'text_body_important')

#actions' frame
features_frame = LabelFrame(root, text = "Actions")
features_frame.pack(fill = 'y', side = RIGHT)

#action buttons
delete_button = Button(features_frame, text = "Hand Bite", command = lambda: collect_data("hand_bite"), width = 15)
delete_button.grid(row = 1, column = 0, padx = 20, pady = 30)

cover_face_butt = Button(features_frame, text = "Cover Face", command = lambda: collect_data("cover_face"), width = 15)
cover_face_butt.grid(row = 3, column = 0, padx = 20, pady = 30)

cover_ears_butt = Button(features_frame, text = "Cover Ears", command = lambda: collect_data("cover_ears"), width = 15)
cover_ears_butt.grid(row = 4, column = 0, padx = 20, pady = 30)

head_scratch_butt = Button(features_frame, text = "Head Scratch", command = lambda: collect_data("head_scratch"), width = 15)
head_scratch_butt.grid(row = 5, column = 0, padx = 20, pady = 30)

no_action_butt = Button(features_frame, text = "No Action", width = 15)
no_action_butt.grid(row = 6, column = 0, padx = 20, pady = 30)

#predict frame button and text
label1 = Label(prediction_frame, text="This button opens the Action Recognition Demo! Note that these action executions will be recorded to be subsequently labeled and used to improve the model.", font= ("Arial", 8))
label2 = Label(prediction_frame, text="If you wish to help and authorize us to collect your data, please check the box. Thanks!", font= ("Arial", 8))
label1.pack()
label2.pack()

#checkbox
checkbox = ttk.Checkbutton (prediction_frame, text = "Yes, I give my permission to collect my data.")
checkbox.pack()

predict_butt = Button(prediction_frame, text = "Predict", command = predict, width = 15)
predict_butt.pack()

root.mainloop()