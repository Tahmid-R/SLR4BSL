
# Sign Language Recognition for British Sign Language

This project aims to develop a sign langauge recognition (SLR) model that can translate British Sign Language (BSL) to English text, providing a means of communication between eharing and non hearing people.

## Brief Overview
We developed this project in Jupyter notebook, using Python version 1.5. The technologies used consist of MediaPipe (detecting hands and extracting landmark data), OpenCV (accessing webcam input for data collection), TensorFlow (building/training model architectures) and more. 

We collected videos and landmark data of BSL words and letters, totalling over 2560 videos.

## The Files Included
Here is what you need to know about project files.

If you want to jump straight ahead and test the SLR model, you should run the python file "SLR4BSL.py". Or, if you want to go through each step of the project, you can view the Jupyter file "SLR4BSL.ipynb". 

There are 3 trained models included, GRU, LSTM and Bi-Directional GRU+GRU; they have been saved to their best weights.

"BSL_DATA.zip" contains all BSL videos and landmark data that we collected. Each participant folder contains 32 sign gesture folders, which includes 10 video folders that store 40 keypoint files and the recorded videos.





## Setting Up Envrionment
To run this project please follow these steps, you will need to install the packages and libraries for it to function.

### 1. Create a Virtual Python Environment (in console window)
We are using Python version 3.10.10. To create a virtual envrionment for a specific version, you need to install virtual env. 

Follow the steps below.

!pip install virtualenv

virtualenv -p "path to python version" "envrionment name"

(activate virtual envrionment)

source "envrionmentname"/bin/activate # Linux

.\"envrionmentname"\Scripts\activate # Windows 



### 2. Setting up Jupyter Notebook 

Install these dependancies to create Python kernel from your virtual environment

python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user  
--name="environmentname"



### 3. Opening Jupyter Notebook and Installing Libraries

Activate your virtual environment and enter "jupyter notebook"

Install these libraries:

!pip install tensorflow==2.14.0 mediapipe==0.10.5 matplotlib opencv-python scikit-learn seaborn gTTS playsound


