#you will need these libraries: !pip install tensorflow==2.14.0 mediapipe==0.10.5 matplotlib opencv-python scikit-learn seaborn gTTS playsound
#gtts and playsound is optional: this feature was removed after user evaluation

import cv2
import mediapipe as mp
import tensorflow
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import random
from tensorflow.keras.models import load_model

    

signs = ['hello','good','sorry','how are you','wait','morning',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
         'O','P','Q','R','S','T','U','V','W','X','Y','Z']

#This is where all data captured will be stored
data_folder = 'MEDIAPIPE_DATA'

#Mediapipe Hands Model
mp_hands = mp.solutions.hands

#Mediapipe Drawing Utils for Drawing Landmarks
mp_drawing_utils = mp.solutions.drawing_utils

#loading saved slr model
model=load_model('slr4bsl-handsgru-5502')



#encapsulating mediapipe detection process into a function - it is repeated often
def mediapipe_detections(hands, frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #making image non writeable helps save space
    image.flags.writeable = False
    
    #start detecting hands in frame
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                            mp_drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    return results, image




def training_mode():
   #store translated word after predicting live data
    translation=None
    
    sign = random.choice(signs)
    #capture 40 frames of keypoints (what model expects) and use for model to predict
    sequence=[]

    model=load_model('slr4bsl-handsgru-5502')
    
    #initalising holistic model
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    cv2.startWindowThread()


    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Empty Frame")
            pass

        try:
            results, image = mediapipe_detections(hands, frame)

            cv2.putText(image,f'Practise {sign}',(250,650),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),3, cv2.LINE_AA)
            cv2.circle(image,(60,100),25,(0,0,255),-1)

            if results.multi_hand_landmarks:
                hand_keypoints = np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[-1].landmark]).flatten()
            else:
                #the model leveraged will expect the same total number of landmarks (63), so we need a zeros array when hands not detected
                hand_keypoints = np.zeros(21*3)
                
            #store keypoints in sequence array
            sequence.append(hand_keypoints)

            #reduce sequence length to 30 values - what the model expects
            sequence = sequence[-40:]

            #model expects a batch dimension so we must put the sequence in one
            pred = model.predict(np.expand_dims(sequence,axis=0))[0]
            translation = signs[np.argmax(pred)]
            print(translation)

            #verifying if user sign gesture is same as sign gesture
            if translation == sign:
                cv2.circle(image,(60,100),25,(0,255,0),-1)
                print('correct')
                sign = random.choice(signs)
            
            #gives option to skip to a different gesture
            if cv2.waitKey(10) & 0xFF==ord('n'):
                sign = random.choice(signs)


        except:
            pass
        cv2.imshow("SLR4BSL Training Mode", image)

         #close video capture logic
        if cv2.waitKey(10) & 0xFF==ord('q'):
            end = True
            break
    

    return end
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

    
def help_mode():
    #the user input is stored to find video filepath of specific sign
    sign = input("Enter sign gesture you need help with: ")
    #important to check if filepath exists - otherwise break
    if os.path.exists(os.path.join(data_folder,'Help Videos',f'{sign}.mp4')):
        
        #optional - but we loop the video 3 times
        for counter in range(3):
            cap = cv2.VideoCapture(os.path.join(data_folder,'Help Videos',f'{sign}.mp4'))
            cv2.startWindowThread()


            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    print("End of Video")
                    break

                cv2.putText(frame,f'{sign}',(450,650),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),3, cv2.LINE_AA)


                cv2.imshow("SLR4BSL Help Mode", frame)

                 #close video capture logic
                if cv2.waitKey(50) & 0xFF==ord('q'):
                    break
    
    #once video is complete or incorrect input, help mode is destroyed
    end = True
    return end
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    
#optional feature - for observational purposes
def switch_models(switch_model):
    if switch_model == False:
        print("Switching to LSTM")
        model = load_model('slr4bsl-handsLSTM-950')
        switch_model = True
           
    elif switch_model:
        print("Switching to GRU")
        model = load_model('slr4bsl-handsgru-5502')
        switch_model = False
    
    return switch_model


def switch_modes(window1,window2,window3):
    #logic of opening different windows (main window, training window and help window)
    
    #shows main window
    if window1:
        cv2.imshow("SLR4BSL", image)
        cv2.destroyWindow("SLR4BSL Training Mode")
        cv2.destroyWindow("SLR4BSL Help Mode")
        
    #shows training mode window
    elif window2:
        endWindow2 = training_mode()
        cv2.destroyWindow("SLR4BSL")
        if endWindow2:
            window1=True
            window2=False
            
    #shows help mode window
    elif window3:
        endWindow3 = help_mode()
        cv2.imshow("SLR4BSL", image)
        if endWindow3:
            window1=True
            window3=False
    
    return window1, window2, window3


#store translated word after predicting live data
translation=None
temp=None

#capture 40 frames of keypoints (what model expects) and use for model to predict
live_sequence=[]

#model = load_model('slr4bsl-handsgru-5502')
#initalising holistic model
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
cv2.startWindowThread()

cv2.namedWindow("SLR4BSL", cv2.WINDOW_NORMAL)

#The first three booleans are used to switch bewteen modes - the default window is set to True
slrWindow = True
trainingWindow = False
helpWindow = False

#This also is used to switch between the GRU and LSTM model
switch_model = False

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Empty Frame")
        pass
    
    try:
        results, image = mediapipe_detections(hands,frame)
        
        if results.multi_hand_landmarks:
            hand_keypoints = np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[-1].landmark]).flatten()
        else:
            #the model leveraged will expect the same total number of landmarks (63), so we need a zeros array when hands not detected
            hand_keypoints = np.zeros(21*3)
        
        #store keypoints in sequence array
        live_sequence.append(hand_keypoints)
        
        #take the last 40 frames from live sequence - what the model expects
        live_sequence = live_sequence[-40:]
        
        #model expects a batch dimension so we must put the sequence in one
        pred = model.predict(np.expand_dims(live_sequence,axis=0))[0]
        temp = translation
        
        #The highest probability in pred is stored to check if it is over 90%
        sign_index = np.argmax(pred)
        
        if sign_index > 0.9:
            translation = signs[sign_index]
            print(translation)
            

        
        #display text output of predicted sign
        cv2.putText(image,f"{translation}",(350,650),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),3, cv2.LINE_AA)
        
        if switch_model == False:
            cv2.putText(image,'GRU',(1000,100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255),3, cv2.LINE_AA)
        elif switch_model:
            cv2.putText(image,'LSTM',(1000,100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255),3, cv2.LINE_AA)

    except:
        pass
    
    #logic of opening different windows (main window, training window and help window)
    slrWindow,trainingWindow, helpWindow = switch_modes(slrWindow,trainingWindow,helpWindow)
        
    
    #close video capture logic
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    #switch to training mode
    elif cv2.waitKey(10) & 0xFF==ord('t'):
        slrWindow = False
        trainingWindow = not trainingWindow
        helpWindow = False
    #switch to help mode
    elif cv2.waitKey(10) & 0xFF==ord('h'):
        slrWindow = False
        trainingWindow = False
        helpWindow = not helpWindow
    elif cv2.waitKey(10) & 0xFF==ord('s'):
        switch_models(switch_model)
           
        
    
    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)