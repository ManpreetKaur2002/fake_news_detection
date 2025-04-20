import os
import pandas as pd
import numpy as np
import re
import nltk
import torch
import gc
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast

# Download NLTK data only if not already present
def download_nltk_data():
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')
    except:
        print("Downloading necessary NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)

download_nltk_data()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Config parameters
MAX_LENGTH = 256  # Reduced from 512 to improve speed
BATCH_SIZE = 16   # Increased from 8 
ACCUMULATION_STEPS = 2  # Gradient accumulation steps
NUM_EPOCHS = 3
WORKERS = 4  # Parallel data loading
LEARNING_RATE = 2e-5
CACHE_DIR = ".cache"
CHECKPOINT_PATH = "bert_fake_news.pth"
VALIDATION_SPLIT = 0.1

# Check and create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare cached preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    return text  # Skip tokenization to speed up preprocessing

def load_and_preprocess_data(cache_path="preprocessed_data.pkl"):
    # Check if preprocessed data exists in cache
    if os.path.exists(os.path.join(CACHE_DIR, cache_path)):
        print(f"Loading preprocessed data from cache...")
        data = pd.read_pickle(os.path.join(CACHE_DIR, cache_path))
        return data['X'], data['y']
    
    print("Loading dataset...")
    df = pd.read_csv("train.csv")
    df.dropna(inplace=True)
    print(f"Dataset loaded with {len(df)} samples.")
    
    print("Preprocessing text...")
    start_time = time.time()
    df['clean_text'] = df['text'].apply(preprocess_text)
    print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds.")
    
    X, y = df['clean_text'].tolist(), df['label'].tolist()
    
    # Cache the preprocessed data
    pd.to_pickle({'X': X, 'y': y}, os.path.join(CACHE_DIR, cache_path))
    print(f"Preprocessed data cached at {os.path.join(CACHE_DIR, cache_path)}")
    
    return X, y

# Dataset class with caching for tokenization
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoding_cache = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Check if encoding is cached
        if idx in self.encoding_cache:
            encoding = self.encoding_cache[idx]
        else:
            encoding = self.tokenizer(
                self.texts[idx], 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            # Cache only occasionally to avoid memory issues
            if idx % 100 == 0:
                self.encoding_cache[idx] = encoding
                
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_data_loaders(X, y, tokenizer, max_length, batch_size, num_workers):
    # Create dataset
    full_dataset = FakeNewsDataset(X, y, tokenizer, max_length)
    
    # Split into train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set: {train_size} samples, Validation set: {val_size} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, checkpoint_path):
    # Initialize mixed precision training if available
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}...")
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixed precision training
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
            
            # Accumulate gradients
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            steps += 1
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Report progress
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
                
            # Clean up memory
            del input_ids, attention_mask, labels, outputs, loss
            if batch_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Make sure to update for the last batch if it doesn't fit evenly
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / steps
        
        # Validation phase
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model improved! Saved checkpoint to {checkpoint_path}")
        
        # Free memory after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Training completed. Best validation F1: {best_val_f1:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1

def main():
    start_time = time.time()
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X, y, tokenizer, MAX_LENGTH, BATCH_SIZE, WORKERS)
    
    # Load model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # Load previous model weights if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading existing model weights from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("Model weights loaded successfully!")
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Train model
    print(f"\nStarting training with {NUM_EPOCHS} epochs...")
    train_model(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS, device, CHECKPOINT_PATH)
    
    # Final evaluation
    print("\nTraining completed. Evaluating on validation set...")
    val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
    print(f"Final validation performance:")
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()