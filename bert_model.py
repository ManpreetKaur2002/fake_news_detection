if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import re
    import nltk
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.metrics import accuracy_score
    from torch.cuda.amp import autocast, GradScaler

    print("Downloading necessary NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    print("Loading dataset...")
    df = pd.read_csv("train.csv")
    df.dropna(inplace=True)
    print(f"Dataset loaded with {len(df)} samples.")

    print("Preprocessing text...")
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'\W', ' ', text)  
        text = re.sub(r'\s+', ' ', text)  
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
        return text

    df['clean_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing completed.")

    print("Extracting features and labels...")
    X, y = df['clean_text'].tolist(), df['label'].tolist()

    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded and moved to device.")

    print("Creating dataset class...")
    class FakeNewsDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            return {**{key: val.squeeze(0) for key, val in encoding.items()}, 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

    print("Creating DataLoader...")
    dataset = FakeNewsDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print("DataLoader created.")

    print("Initializing optimizer and loss function...")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Optimizer and loss function initialized.")

    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    print("Starting training loop...")
    model.train()
    for epoch in range(3):
        print(f"Starting epoch {epoch+1}...")
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader)}")

    print("Training completed. Starting evaluation...")
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            y_pred.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            if batch_idx % 10 == 0:
                print(f"Evaluating batch {batch_idx}/{len(train_loader)}")

    print("Evaluation completed.")
    print("BERT Accuracy:", accuracy_score(y, y_pred))
