import os
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling warnings for gpu requirements

import nltk
# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

import pickle
from pickle import load
import json 
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
#from keras.models import load_model
from tensorflow.python.keras.models import load_model

#from keras_preprocessing.sequence import pad_sequences
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D
try:
    print("Uploading....")
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/intents.json") as file:
        intents= json.load(file)
        print("JSON file upload done")
except FileNotFoundError as e:
    print(e)

try:
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/tokenizer.pickle", "rb") as file_tokenizer:
        tokenizer = pickle.load(file_tokenizer)
except FileNotFoundError as e:
    print(f"Error: Tokenizer file not found - {e}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Ensure that the file format and content match the expected tokenizer structure.")
else:
    print("Tokenizer loaded successfully.")
#loading label encoder
try:
    with open(r"/home/kg/project/voice-assistant-system-candy/Data/label_encoder.pickle","rb") as enc:
        lbl_encoder=pickle.load(enc)
except Exception as e:
    print(e)

import tensorflow as tf    
#loading the saved model
try:
    # load trained model
    model = load_model("/home/kg/project/voice-assistant-system-candy/Data/model_trained.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")




#loading the tokenizer

 
#prediction fucntion    
def predict_intent_with_nltk(model, tokenizer, label_encoder, text, max_len, ps, stop_words):
    # Tokenize, remove stop words, and apply stemming
    words = nltk.word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    processed_text = ' '.join(words)
    print(processed_text)
    # Tokenize and pad the processed input text
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)

    # Make the prediction using the trained model
    result = model.predict(padded_sequence, verbose=False)

    # Convert the model's output to the predicted intent
    predicted_intent = label_encoder.inverse_transform([np.argmax(result)])[0]

    return predicted_intent
def response_generator():
    pass

# Example usage:
max_len=20
text_to_predict = "hi candy can you recommend me  movies"
text_len=min(max_len,len(text_to_predict))
predicted_intent = predict_intent_with_nltk(model, tokenizer, lbl_encoder, text_to_predict, text_len, ps, stop_words)
print(f"Predicted Intent: {predicted_intent}")
