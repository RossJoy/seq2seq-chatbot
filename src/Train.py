import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold

# Assuming 'model' is a GPT-2 model from Hugging Face
# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Assuming 'train_loader' and 'test_loader' are your DataLoader instances for training and testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model, train_loader, test_loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        # Training phase
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        average_loss = total_loss / len(train_loader)
        print(f"Average Loss after epoch {epoch+1}: {average_loss:.4f}")

        # Validation phase
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_ids'].to(device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                total_loss += loss.item()

        average_loss = total_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(average_loss)).to(device)
        print(f'Perplexity after epoch {epoch+1}: {perplexity.item()}')

# Usage
train(model, train_loader, test_loader, optimizer, epochs)
