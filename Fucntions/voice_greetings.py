import warnings

# Suppress InconsistentVersionWarning from scikit-learn
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import random


file_path_1 = r"C:\Users\kashy\Documents\project\project\chatbot_model.joblib"
loaded_model = joblib.load(file_path_1)

# Load the fitted vectorizer
file_path_2 = r"C:\Users\kashy\Documents\project\project\vectorizer.joblib"
vectorizer = joblib.load(file_path_2)
file_path_3 = r"C:\Users\kashy\Documents\project\project\intents.json"
try:
    with open(file_path_3, 'r') as file:
        intents= json.load(file)
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")

# Function for making predictions
def predict_intent(text):
    text=text.lower()
    input_vector=vectorizer.transform([text])
    intent=loaded_model.predict(input_vector)[0]
    
    if intent in intents:
        responses = intents[intent]['responses']
        response = random.choice(responses)
        print("intent : ",intent)
        print("Response : ",response)
        return [intent,response]
        

predict_intent("HI")

    
