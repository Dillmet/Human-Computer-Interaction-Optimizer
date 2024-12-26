import streamlit as st
import pyttsx3
import speech_recognition as sr
from streamlit_option_menu import option_menu
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math



# Sidebar for navigation
with st.sidebar:
    st.image('Capture.PNG', use_column_width=True)
    selected = option_menu('COMMUNICATION SITE FOR SPECIALLY ABLED',
                           ['Text to Speech',
                            'Speech to Text',
                            'Text to Sign',
                            'Sign to Text',
                            'Help'],
                           icons=['heart', 'heart', 'heart', 'heart', 'person'],
                           default_index=0)
    if selected == 'Text to Speech':
        # Dropdown menu for default sentences
        default_sentences = ['Hello, how are you?',
                             'What is the weather today?',
                             'Can you please help me?',
                             'Thank you for your assistance.']
        default_sentence = st.selectbox("Choose a default sentence:", default_sentences)

# Function to perform speech-to-text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

    try:
        st.write("Transcribing...")
        text = r.recognize_google(audio_data=audio)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        st.write("Sorry, Google Speech Recognition service is down.")

# Function to perform text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to convert text to sign language gestures
def text_to_sign(text):
    # Dummy function to generate sign language gestures (replace with your own implementation)
    # For demonstration purpose, let's just return the input text
    return text

# Function to convert sign language gestures to text
def sign_to_text():
    labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("keras_model.h5" , "labels.txt")
    offset = 20
    imgSize = 300

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)

            return labels[index]

# Text to Speech Page
if selected == 'Text to Speech':
    # Page title
    st.title('Text to Speech')

    # Text-to-speech option
    text_input = st.text_input("Enter text to convert to speech:")
    if st.button("Convert to Speech"):
        if text_input:
            text_to_speech(text_input)
        else:
            st.write("Please enter some text.")

# Speech to Text Page
elif selected == 'Speech to Text':
    # Page title
    st.title('Speech to Text')

    # Speech-to-text option
    if st.button("Record"):
        speech_text = speech_to_text()
        if speech_text: 
            st.write("Transcribed Speech:", speech_text)

# Text to Sign Page
elif selected == 'Text to Sign':
    # Page title
    st.title('Text to Sign')

    # Text-to-sign option
    text_input = st.text_input("Enter text to convert to sign language:")
    if st.button("Convert to Sign"):
        if text_input:
            sign_text = text_to_sign(text_input)
            st.write("Sign Language Gestures:", sign_text)
        else:
            st.write("Please enter some text.")

# Sign to Text Page
elif selected == 'Sign to Text':
    # Page title
    st.title('Sign to Text')

    # Sign-to-text option
    st.write("Showing Predicted Text from Sign Language")
    sign_text = sign_to_text()
    st.write("Predicted Text:", sign_text)

# Help Page
elif selected == 'Help':
    # Page title
    st.title('Help')
    st.write("If you need any help with something feel free to connect to us through the given helpline number")
    st.write("Contact :", 9898989898)