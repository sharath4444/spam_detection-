import pandas as pd
import re
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('Spam_Data.csv')
df.columns = df.columns.str.strip()  # Remove whitespace from column names
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})  # Convert labels to binary

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

# Apply preprocessing
df['CleanMessage'] = df['Message'].apply(preprocess_text)

# CountVectorizer with N-Grams (Unigrams, Bigrams, Trigrams)
vectorizer_ngram = CountVectorizer(ngram_range=(1, 3))
X = vectorizer_ngram.fit_transform(df['CleanMessage'])
y = df['Category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Initialize Models
logistic_regression = LogisticRegression()
naive_bayes = MultinomialNB()

# Train Models
logistic_regression.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predict
y_pred_lr = logistic_regression.predict(X_test)
y_pred_nb = naive_bayes.predict(X_test)

# Evaluate
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Logistic Regression Accuracy:", accuracy_lr)
print("Naive Bayes Accuracy:", accuracy_nb)


# Save the models & vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer_ngram, f)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(logistic_regression, f)

with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(naive_bayes, f)

print("✅ Models and vectorizer saved successfully!")
