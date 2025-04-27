import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import nltk
import re

# Preload Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Precompile regexes for faster processing
# Precompile regex patterns for URLs and special characters
url_pattern = re.compile(r'https?://\S+')
special_char_pattern = re.compile(r'[^a-z\s]')

# 1. Load dataset
print("✅ CSV found, loading...")
data = pd.read_csv('mbti_1.csv')

# 2. Clean text much faster
print("✅ Cleaning text (superfast)...")

def clean_text_fast(text):
    text = text.lower()
    text = url_pattern.sub('', text)
    text = special_char_pattern.sub('', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning with faster processing
data['posts'] = data['posts'].apply(clean_text_fast)

# 3. Prepare Inputs
print("✅ Vectorizing...")
X = data['posts']
y = data['type']

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 5. Model
print("✅ Training model...")
model = RandomForestClassifier()
model.fit(X_vec, y)     #Train it: model.fit(features, labels).

# 6. Save Model and Vectorizer
print("✅ Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("🎉 All Done! Model and vectorizer saved successfully.")
