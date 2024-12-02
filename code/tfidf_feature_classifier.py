import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')

def get_top_tfidf_words(tfidf_matrix, feature_names, label_indices, n_words=1000):
    """Get top N words based on TF-IDF scores for a specific label."""
    # Get the average TF-IDF score for each word across documents of this label
    avg_tfidf = np.mean(tfidf_matrix[label_indices].toarray(), axis=0)
    # Get indices of top words
    top_indices = np.argsort(avg_tfidf)[-n_words:]
    print("debug: ")
    
    return [feature_names[i] for i in top_indices]

def create_binary_features(text, important_words):
    """Create binary features indicating presence/absence of important words."""
    tokens = set(word_tokenize(text.lower()))
    return {word: (word in tokens) for word in important_words}

# Read the first 2000 rows
print("Reading training data...")
df = pd.read_csv('data/train.csv', nrows=20000)
df['description'] = df['description'].fillna('')

# Initialize and fit TF-IDF vectorizer
print("Calculating TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
feature_names = tfidf.get_feature_names_out()

# Get top words for each label
print("Extracting top words for each label...")
important_words = set()

print(len(df['label']))
for label in df['label'].unique():
    label_indices = df['label'] == label
  
    top_words = get_top_tfidf_words(tfidf_matrix, feature_names, label_indices)
    important_words.update(top_words)

print(f"Total unique important words: {len(important_words)}")

# Create binary features for each description
print("Creating binary features...")
X = []
for text in df['description']:
    features = create_binary_features(text, important_words)
    X.append([int(v) for v in features.values()])

X = np.array(X)
y = df['label']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
print("Training classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate
print("\nEvaluation on validation set:")
y_pred = classifier.predict(X_val)
print(classification_report(y_val, y_pred))

# Function to predict sentiment for new text
def predict_sentiment(text):
    features = create_binary_features(text, important_words)
    feature_vector = np.array([int(v) for v in features.values()]).reshape(1, -1)
    prediction = classifier.predict(feature_vector)[0]
    probability = classifier.predict_proba(feature_vector)[0]
    return prediction, probability

# Save important words and model for future use
import pickle
print("\nSaving model and features...")
model_data = {
    'important_words': list(important_words),
    'classifier': classifier
}
with open('code/tfidf_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Example usage
print("\nTesting with example sentences:")
test_sentences = [
    "This product is amazing and works perfectly!",
    "Terrible quality, would not recommend.",
    "Average product, does the job but nothing special."
]

for sentence in test_sentences:
    prediction, probabilities = predict_sentiment(sentence)
    print(f"\nText: {sentence}")
    print(f"Predicted label: {prediction}")
    print(f"Confidence scores: {probabilities}") 