import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load models and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer_ngram = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    logistic_regression = pickle.load(f)

with open("naive_bayes_model.pkl", "rb") as f:
    naive_bayes = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("This application detects spam messages using **Logistic Regression & Naive Bayes** with **N-Grams**.")

# Input text for prediction
user_input = st.text_area("Enter a message to check if it's spam or not:")

if st.button("Check Spam"):
    if user_input:
        input_vectorized = vectorizer_ngram.transform([preprocess_text(user_input)])
        
        pred_lr = logistic_regression.predict(input_vectorized)[0]
        pred_nb = naive_bayes.predict(input_vectorized)[0]

        result_lr = "Spam" if pred_lr == 1 else "Not Spam"
        result_nb = "Spam" if pred_nb == 1 else "Not Spam"

        st.subheader("Prediction Results:")
        st.write(f"🔹 **Logistic Regression:** {result_lr}")
        st.write(f"🔹 **Naive Bayes:** {result_nb}")

    else:
        st.warning("Please enter a message!")
