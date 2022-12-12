# AI_TOP
The project AI TOP focuses on using AI to help students with disabilities by recognizing signs of distress and predicting crisis.

For now there are only some files and those may seem like a mess, but I will soon upload the rest of the scripts, and will keep the repo up to date and tidy.

###### Current weights
So far, the best weights are stored in: 5_actions_0.844.h5
Be sure to use the right weights in your file!

First, you have to create the virtual environment! 
Make sure to have Anaconda installed, then open the Anaconda Terminal (from search bar).
Once you are in your Anaconda terminal, navigate to the repository folder in your workstation (where all the files and the conda_environment.yml is) and type: 
##### conda env create -f conda_environment.yml
this command will create the environment (that is called mediapipe1). Once it is done, type 
##### conda activate mediapipe1
now you will see the terminal line will start with (mediapipe1), this means you are inside the environment. to exit the environment type:
##### conda deactivate

Once you are in, navigate to the repository's folder (you should already be there but also for future use) and type:
##### python Demo.py
or
##### python GUI.py
depending on what file you want to open, that's it!

## Demo.py
This is a python file that opens a GUI that allows to try the real-time action recognition demo. For now it only recognizes: no action, hand biting, head scratching, covering face and covering ears, but it will be soon expanded. If the checkbox in the GUI is checked, the software will record videos that will be found in a folder called data_to_label. If you wish to help the cause, you are free to collect the data and then contact me so that we can get in touch and use your data for research.

## GUI.py
This one is a more elaborate version of the demo, it opens a gui where one can choose to predict or to record any of the listed actions. If you wish to record the actions, follow the protocol explained in the file.

## variable_sequence_length_LSTM.ipynb
Using this notebook one can train a novel model based on the data we collected so far, and you can also add new data to the DATA folder, following the convention used (a folder for action that contains a folder for each sequence). Trained models can also be tested immediately!

## functions.py
The functions file is a set of functions used by the other files, so make sure to have them both in the same folder or nothing will run!

# Thanks for your time!
If you have any questions, ideas or feedback don't hesitate to contact me, thanks and enjoy!
