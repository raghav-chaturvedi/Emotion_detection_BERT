#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 00:06:31 2025

@author: raghavchaturvedi
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text(text, tokenizer, max_len=128):
    """
    Preprocesses a given text by tokenizing it, adding special tokens, 
    padding to a fixed length, and truncating if necessary.
    
    Args:
        text (str): The input text to preprocess.
        tokenizer: The tokenizer instance for BERT.
        max_len (int): The maximum length for tokenized sequences.

    Returns:
        tuple: Flattened input_ids and attention_mask tensors.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()

train = pd.read_csv("train.csv")
train_text = train["text"]
train_label = train["label"]

test = pd.read_csv("test.csv")
test_id = test["id"]
test_text = test["text"]

input_ids, attention_mask = preprocess_text(train_text[0], tokenizer)
#print(input_ids, attention_mask)

# Define datasets

class TextDataset(Dataset):
    """
    Custom Dataset class for text data, compatible with PyTorch DataLoader.
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        Initializes the dataset with text, labels, and tokenizer.
        
        Args:
            texts (list): List of input texts.
            labels (list): List of corresponding labels (or None for test data).
            tokenizer: BERT tokenizer for text preprocessing.
            max_len (int): Maximum length for tokenized sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized input, attention mask, and label for a given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: Contains input_ids, attention_mask, and label tensors.
        """
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else -1

        input_ids, attention_mask = preprocess_text(text, self.tokenizer, self.max_len)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
# Split data into train and validation datasets

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_text, train_label, test_size=0.2, random_state=42
)

# Reset indices after splitting
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# Creating Dataset and DataLoader instances for training, validation, and testing
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_text, None, tokenizer)

# DataLoader to fetch batches of data for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Attention Layer

class AttentionLayer(nn.Module):
    """
    Custom attention mechanism to weigh hidden states based on learned importance scores.
    """
    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): Size of the hidden state from the BERT model.
        """
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))
    
    def forward(self, hidden_states):
        """
        Compute attention scores and weighted sum of hidden states.
        
        Args:
            hidden_states (Tensor): Hidden states from the model (batch_size, seq_len, hidden_size).
        
        Returns:
            Tensor: Weighted sum of hidden states (batch_size, hidden_size).
        """
        # hidden_states: [batch_size, seq_len, hidden_size]
        attention_scores = torch.matmul(hidden_states, self.attention_weights)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
    
        weighted_sum = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]    
        return weighted_sum
    
# Define the model

class TextClassifier(nn.Module):
    """
    Text classification model combining BERT, attention, and a linear classifier.
    """
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') # Pretrained BERT model
        self.dropout = nn.Dropout(0.3) # Dropout for regularization
        self.attention = AttentionLayer(self.bert.config.hidden_size) # Attention layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes) # Classifier

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids (Tensor): Tokenized input IDs (batch_size, seq_len).
            attention_mask (Tensor): Attention mask (batch_size, seq_len).
        
        Returns:
            Tensor: Logits for each class (batch_size, num_classes).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        enhanced_output = self.attention(hidden_states)  # [batch_size, hidden_size]      
        logits = self.classifier(enhanced_output)
        return logits
    
# Initialize the model, optimizer, and loss function

num_classes = len(train_label.unique()) # Number of unique labels in the training set
model = TextClassifier(num_classes).to(device) # Model
optimizer = optim.AdamW(model.parameters(), lr=2e-5) # Define optimizer - use AdamW and set learning rate as 2e-5
loss_fn = nn.CrossEntropyLoss() # Define loss function - CrossEntropyLoss

best_model_state = None # To store the best-performing model state during validation
best_accuracy = 0 # To track the highest validation accuracy achieved
epochs = 3 # Define the number of epochs - can be larger but due to time constraint

## Train
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad() # Clear previous gradients

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() # Accumulate the loss

        loss.backward() # Backward pass to compute gradients
        optimizer.step() # Update model parameters

    avg_loss = total_loss / len(train_loader) # Compute average loss for the epoch
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

## Validation
model.eval()
predictions, true_labels = [], []

with torch.no_grad(): # Disable gradient computation for evaluation
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.to(device).numpy())
        true_labels.extend(labels.to(device).numpy())

accuracy = accuracy_score(true_labels, predictions) # Calculate validation accuracy
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the best model's state
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model_state = model.state_dict() 
    
num_classes = len(train_label.unique())
model_test = TextClassifier(num_classes).to(device)

# Check the model
if best_model_state is not None:
    model_test.load_state_dict(best_model_state)
    print("yes")
else:
    print("no")
    
## Test
def predict(model, data_loader, device):
    """
    Predicts the labels for the given data using the trained model.
    
    Args:
        model: Trained model to use for predictions.
        data_loader: DataLoader for test data.
        device: Device to run the predictions on (e.g., GPU or CPU).

    Returns:
        list: Predicted labels for the test data.
    """
    model.eval()
    predictions = []

    with torch.no_grad(): # Disable gradient computation
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.to(device).numpy())

    return predictions

# Generate predictions for the test set
test_predictions = predict(model_test, test_loader, device)

# Check the prediction
print(test_predictions[:10])  
#print(test_predictions.shape)

# Generate and save the prediction file
submission = pd.DataFrame({'id': id, 'label': test_predictions}) # Combine the id the label prediction
submission.to_csv('submission_bert.csv', index=False) # Save the file
