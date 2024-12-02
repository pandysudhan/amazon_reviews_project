import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk

# Download required NLTK data
nltk.download('punkt')

# Read the first 2000 rows of training data
print("Reading training data...")
df = pd.read_csv('../data/train.csv', nrows=20000)

# Tokenize descriptions
print("Tokenizing descriptions...")
df['tokenized_description'] = df['description'].fillna('').apply(word_tokenize)

# Train Word2Vec model
print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=df['tokenized_description'], 
                    vector_size=100,  # embedding dimension
                    window=5,
                    min_count=1,
                    workers=4)

# Function to get embedding for a single description
def get_description_embedding(tokens, model):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Generate embeddings for all descriptions
print("Generating embeddings for descriptions...")
X = np.array([get_description_embedding(tokens, w2v_model) for tokens in df['tokenized_description']])
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