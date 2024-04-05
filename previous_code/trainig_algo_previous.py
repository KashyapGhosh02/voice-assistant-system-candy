import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Download NLTK resources (tokenizer)
nltk.download('punkt')

# Load intents from JSON file in read mode
file_path = r"c:\Users\kashy\Documents\project\project\Data\intents.json"

try:
    with open(file_path, 'r') as file:
        intents = json.load(file)
        print(type(intents))
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")

# Train the model
training_data = []
labels = []

for intent, data in intents.items():
    for pattern in data['patterns']:
        tokens = word_tokenize(pattern.lower())  # Tokenize the text
        training_data.append(' '.join(tokens))   # Join tokens back into a string
        labels.append(intent)

Vectorizer = TfidfVectorizer()
X_train = Vectorizer.fit_transform(training_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, labels, test_size=0.4, random_state=42, stratify=labels)

model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and vectorizer
joblib.dump(Vectorizer, r"Data\vectorizer.joblib")
joblib.dump(model, r"Data\chatbot_model.joblib")