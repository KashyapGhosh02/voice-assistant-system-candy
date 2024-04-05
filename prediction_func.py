import os
import logging
import random
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling warnings for gpu requirements

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
import json 
try:
    # print("Uploading....")
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/intents.json") as file:
        intents= json.load(file)
        # print("JSON file upload done")
except FileNotFoundError as e:
    print(e)

from keras_preprocessing.sequence import pad_sequences
import numpy as np
#from keras.models import load_model
from pickle import load
from tensorflow.python.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

import tensorflow as tf 

#loading the tokenizer

from keras.preprocessing.text import tokenizer_from_json
import json

# Load the tokenizer
loaded_tokenizer = None
try:
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/tokenizer.json", 'r', encoding='utf-8') as json_file:
        loaded_tokenizer_json = json_file.read()
        loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)
    # print("Tokenizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Tokenizer file not found - {e}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    
# print(type(loaded_tokenizer))
#loading label encoder
try:
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/label_encoder.pickle","rb") as enc:
        lbl_encoder=load(enc)
except FileNotFoundError as e:
    print(e)
    
#loading the saved model
#loading the saved model
try:
    # load trained model
    model = load_model("/home/kg/project/voice-assistant-system-candy/Data/model_trained.h5")
    # print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
 
#prediction fucntion    
def predict_intent_with_nltk(text,model=model, tokenizer=loaded_tokenizer, label_encoder=lbl_encoder, max_len=15, ps=ps, stop_words=stop_words):
    # Tokenize, remove stop words, and apply stemming
    # words = nltk.word_tokenize(text.lower())
    # words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    # processed_text = ' '.join(words)
    # print(processed_text)
    # Tokenize and pad the processed input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)

    # Make the prediction using the trained model
    result = model.predict(padded_sequence, verbose=False)

    # Convert the model's output to the predicted intent
    predicted_intent = label_encoder.inverse_transform([np.argmax(result)])[0]
    
    if predicted_intent in intents:
        responses = intents[predicted_intent]['responses']
        response = random.choice(responses)
        #print("intent : ",predicted_intent)
        #print("Response : ",response)
        
        return [predicted_intent,response]

    # return predicted_intent
def response_generator():
    pass

# Example usage:
# text_to_predict = " i am very sorry"
# text_len=len(text_to_predict)
# predicted_intent = predict_intent_with_nltk(text=text_to_predict)
# print(f"Predicted Intent: {predicted_intent}")
