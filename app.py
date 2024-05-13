from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and other necessary data
model = load_model('chatbot_model_5.h5')
intents = json.loads(open('intents.json', encoding="utf8").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Function to clean up the sentences
def clean_up_sentence(sentence):
    ignore_words = ['?', '!', '@', '$']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_words]
    return sentence_words

# Function to convert sentence into bag of words
def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    print(sentence_words)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the intent of the sentence
def predict_intent(sentence, model, words, classes):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.75
    # choose the results with accuracy which indicate the matching between the question and tag greater than 75% 
    results = ((i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD)   
    # sort from greatest to lowest 
    results = sorted(results, key=lambda x: x[1], reverse=True)
    # put all into a list which contain the tag and their probability
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Function to get response based on intent
def get_response(intents_list, intents_data):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_data['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

# Function to get chatbot response
def chatbot_response(message):
    intents_list = predict_intent(message, model, words, classes)
    response = get_response(intents_list, intents)
    return response

# Route for the home page
@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Chatbot API!"})

# Route to handle chatbot requests
@app.route("/chatbot", methods=['POST'])
def chatbot():
    data = request.json  # Extract JSON data from the request
    if 'message' in data:
        message = data['message']  # User input from the frontend
        response = chatbot_response(message)  # Get the response from the chatbot
        return jsonify({"response": response})
    else:
        return jsonify({"error": "No message provided."})

if __name__ == '__main__':
    app.run(debug=True)
