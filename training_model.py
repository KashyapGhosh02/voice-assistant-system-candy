#importing libaies 

import json
import numpy as np
import os
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

import tensorflow as tf  

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# load the json file 
try:
    print("Uploading....")
    with open(r'Data\intents.json') as file:
        intents= json.load(file)
        print("JSON file upload done")
except FileNotFoundError as e:
    print(e)
    

#append the data into sentences and labels

training_sentences = []
training_labels = []
labels = []
responses = []

for intent,data in intents.items():
    for pattern in data['patterns']:
        training_sentences.append(pattern.lower())
        training_labels.append(intent)
    # for response in data['responses']:
    #     responses.append(response.lower())
    if intent not in labels:
        labels.append(intent)   
   
#print(training_sentences)
#print(training_labels)
#print(labels)
# print(responses)


#train the model 
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# accuracy of the model
accuracy = history.history['accuracy'][-1]  # Get the accuracy from the last epoch

print(f"Final Accuracy: {accuracy * 100:.2f}%")

import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__



# saving the model in local computer 
try:
     #tf.keras.models.save_model(model, r'Data\chat_model')
     model.save(r'Data\model_trained.h5')
     print("model has been saved in local computer")
except Exception as e:
     print(f"Error during model saving: {e}")


import pickle

# to save the fitted tokenizer
try:
    with open(r'Data\tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e :
    print(e)


# to save the fitted label encoder
try :
    with open(r'Data\label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e :
    print(e)
    

# import tensorflow as tf
# # print(f"TensorFlow version: {tf.__version__}")
# # print(f"Keras version: {tf.keras.__version__}")

#prediction fucntion    
# def predict_intent_with_nltk(model, tokenizer, label_encoder, text, max_len, ps, stop_words):
#     # Tokenize, remove stop words, and apply stemming
#     words = nltk.word_tokenize(text.lower())
#     words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
#     processed_text = ' '.join(words)
#     print(processed_text)
#     # Tokenize and pad the processed input text
#     sequence = tokenizer.texts_to_sequences([processed_text])
#     padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)

#     # Make the prediction using the trained model
#     result = model.predict(padded_sequence, verbose=False)

#     # Convert the model's output to the predicted intent
#     predicted_intent = label_encoder.inverse_transform([np.argmax(result)])[0]

#     return predicted_intent
# def response_generator():
#     pass

# Example usage:
#text_to_predict = "can you recommend me  movies"
#text_len=len(text_to_predict)
#predicted_intent = predict_intent_with_nltk(model, tokenizer, lbl_encoder, text_to_predict, text_len, ps, stop_words)
#print(f"Predicted Intent: {predicted_intent}")

