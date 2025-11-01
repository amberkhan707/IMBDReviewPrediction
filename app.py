import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import numpy as np

st.title("IMBD Review Prediction")

user_review = st.text_input('Write Your Review')

model = load_model('SimpleRNNmodel.h5')

# Reverse the index with words to convert user text into vector
word_to_index  = imdb.get_word_index()
index_to_word = {value:key for key,value in word_to_index.items()}

# Conversion of User Review in vector and prediction of sentiment of user
def predict_sentiment(review):
    review.lower().split()
    numerical_review = [word_to_index.get(word,-2) + 3 for word in review]
    numerical_review = pad_sequences([numerical_review],500)
    prediction = model.predict(numerical_review)
    if prediction[0][0] < 0.5 : return "negative",prediction[0][0]
    if prediction[0][0] >= 0.5 : return "positive",prediction[0][0]

# Predcition Value and sentiment of the User review 
Sentiment,Score  = predict_sentiment(user_review)

st.header(f"Sentiment of the reviewer is {Sentiment} with accuracy {Score}")