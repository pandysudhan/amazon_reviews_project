import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')

def get_bert_embedding(text, tokenizer, model):
    # Tokenize and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]

# Read the first 2000 rows of training data
print("Reading training data...")
df = pd.read_csv('../data/train.csv', nrows=2000)

# Initialize BERT
print("Loading BERT model...")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Word2Vec Processing
print("\n=== Word2Vec Processing ===")
print("Tokenizing descriptions...")
df['tokenized_description'] = df['description'].fillna('').apply(word_tokenize)

print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=df['tokenized_description'], 
                    vector_size=100,
                    window=5,
                    min_count=1,
                    workers=4)

def get_w2v_embedding(tokens, model):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Generate Word2Vec embeddings
print("Generating Word2Vec embeddings...")
X_w2v = np.array([get_w2v_embedding(tokens, w2v_model) for tokens in df['tokenized_description']])

# Generate BERT embeddings
print("\n=== BERT Processing ===")
print("Generating BERT embeddings...")
X_bert = np.array([get_bert_embedding(text, bert_tokenizer, bert_model) 
                   for text in tqdm(df['description'].fillna(''))])

y = df['label']

# Evaluate Word2Vec
print("\n=== Word2Vec Results ===")
X_train_w2v, X_val_w2v, y_train, y_val = train_test_split(X_w2v, y, test_size=0.2, random_state=42)
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = clf_w2v.predict(X_val_w2v)
print("\nWord2Vec Classification Report:")
print(classification_report(y_val, y_pred_w2v))

# Evaluate BERT
print("\n=== BERT Results ===")
X_train_bert, X_val_bert, y_train, y_val = train_test_split(X_bert, y, test_size=0.2, random_state=42)
clf_bert = LogisticRegression(max_iter=1000)
clf_bert.fit(X_train_bert, y_train)
y_pred_bert = clf_bert.predict(X_val_bert)
print("\nBERT Classification Report:")
print(classification_report(y_val, y_pred_bert)) 