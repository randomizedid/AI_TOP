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
checkbox = ttk.Checkbutton (prediction_frame, text = "Yes, I give my permission to collect my data.")
checkbox.pack()

#Setting up the predict button.
predict_butt = Button(prediction_frame, text = "Start the Demo!", command = lambda: predict(checkbox, model, actions), width = 15)
predict_butt.pack()

#Running the Tkinter loop that handles and updates the GUI.
root.mainloop()