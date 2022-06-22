# AI_TOP
The project AI TOP focuses on using AI to help students with disabilities by recognizing signs of distress and predicting crisis.

For now there are only some files and those may seem like a mess, but I will soon upload the rest of the scripts, and will keep the repo up to date and tidy.

###### Current weights
In this section I will keep the name of the best model weights I have been able to train, be sure to check your version of the file is using the right one!
So far, the best weights are stored in: 5_actions_0.844.h5

Let's see what the file do!

## Demo.py
This is a python file that opens a GUI that allows to try the real-time action recognition demo. For now it only recognizes: no action, hand biting, head scratching, covering face and covering ears, but it will be soon expanded. If the checkbox in the GUI is checked, the software will record videos that will be found in a folder called data_to_label. If you wish to help the cause, you are free to collect the data and then contact me so that we can get in touch and use your data for research.

## GUI.py
This one is a more elaborate version of the demo, it opens a gui where one can choose to predict or to record any of the listed actions. If you wish to record the actions, follow the protocol explained in the file.

## variable_sequence_length_LSTM.ipynb
Using this notebook one can train a novel model based on the data we collected so far, and you can also add new data to the DATA folder, following the convention used (a folder for action that contains a folder for each sequence). Trained models can also be tested immediately!

## functions.py
The functions file is a set of functions used by the other files, so make sure to have them both in the same folder or nothing will run!

#Thanks for your time!
If you have any questions, ideas or feedback don't hesitate to contact me, thanks and enjoy!
