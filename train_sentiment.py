import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from collections import Counter
import numpy as np

# Load Tweets data
df = pd.read_csv("Tweets.csv")
print(f"Loaded {len(df)} tweets")

# Filter to only sentiment-labeled data
df = df[df['airline_sentiment'].notna() & df['text'].notna()]
print(f"After filtering: {len(df)} tweets")

# Map sentiments to labels: 0=Negative, 1=Neutral, 2=Positive
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(sentiment_map)
df = df[df['label'].notna()]
print(f"Valid labels: {len(df)} tweets")

def apply_negation_handling(tokens):
    """Handle negation by modifying tokens after negation words."""
    negation_words = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", 
                      "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", 
                      "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mightn't"}
    modified_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i + 1 < len(tokens):
            modified_tokens.append(tokens[i])
            modified_tokens.append("NOT_" + tokens[i + 1])
            i += 2
        else:
            modified_tokens.append(tokens[i])
            i += 1
    return modified_tokens

def preprocess_text(text):
    """Clean and tokenize text with negation handling."""
    # Remove URLs, @mentions, special chars
    text = text.lower()
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    tokens = text.split()
    tokens = [t.strip('.,!?;:') for t in tokens if t.strip('.,!?;:')]
    tokens = apply_negation_handling(tokens)
    return tokens

# Build vocabulary from training data
print("\nBuilding vocabulary with negation-aware tokens...")
all_tokens = []
for text in df['text']:
    tokens = preprocess_text(text)
    all_tokens.extend(tokens)

# Create vocab with min frequency threshold
token_counts = Counter(all_tokens)
min_freq = 2
vocab = {'<PAD>': 0, '<UNK>': 1}
idx = 2
for token, count in token_counts.most_common():
    if count >= min_freq:
        vocab[token] = idx
        idx += 1

print(f"Vocabulary size: {len(vocab)} (includes negation-aware tokens)")
print(f"Sample negated tokens: {[k for k in vocab.keys() if k.startswith('NOT_')][:10]}")

# Prepare data
max_len = 30
X = []
y = []

for idx_row, row in df.iterrows():
    tokens = preprocess_text(row['text'])
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        else:
            indices.append(vocab['<UNK>'])
    
    # Pad or truncate
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    X.append(indices)
    y.append(row['label'])

X = torch.LongTensor(X)
y = torch.LongTensor(y)
print(f"\nDataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y.numpy())}")

# Create dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model (same architecture as app.py)
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Initialize model
vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
output_dim = 3

model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_dim)
device = torch.device('cpu')
model = model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

print(f"\nTraining model for {epochs} epochs...")
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

# Save model and vocabulary
print("\nSaving model and vocabulary...")
torch.save(model.state_dict(), "models/sentiment_model.pth")
pickle.dump(vocab, open("models/sentiment_vocab.pkl", "wb"))
print("✓ Model saved to models/sentiment_model.pth")
print("✓ Vocabulary saved to models/sentiment_vocab.pkl")
print("\nYour sentiment model is now trained with negation awareness!")
print("Test it in the app with: 'wow my flight is not late with this airline!'")
