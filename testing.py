print("Loading test dataset...")
df_test = pd.read_csv("test.csv").dropna()
print(f"Test dataset loaded with {len(df_test)} samples.")

# Extract text
X_test = df_test['text'].tolist()

# Define FakeNewsDataset for testing (no labels)
print("Creating test dataset...")
class FakeNewsDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}

# Create test DataLoader
test_dataset = FakeNewsDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Starting predictions...")
model.eval()
predictions = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_labels = torch.argmax(outputs, axis=1).cpu().numpy()
        predictions.extend(pred_labels)

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx}/{len(test_loader)} batches.")

print("Predictions completed.")

# Save results
df_test['predicted_label'] = predictions
df_test[['id', 'predicted_label']].to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv successfully!")
