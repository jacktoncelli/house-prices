import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
import os
import datetime

from dataset import HousePriceDataset
from neuralnet import NeuralNetHousePrice

# Hyper parameters
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
output_size = 1
num_epochs = 500
batch_size = 8
learning_rate = 0.003
dropout_prob = 0.1

data_dir = os.path.join(os.getcwd(), "data")
train_csv = os.path.join(data_dir, "cleaned_train.csv")
test_csv = os.path.join(data_dir, "cleaned_test.csv")
validation_csv = os.path.join(data_dir, "cleaned_val.csv")

# Load data
train_dataset = HousePriceDataset(dataset_path=train_csv, dataset_type="train")
test_dataset = HousePriceDataset(dataset_path=test_csv, dataset_type="test")
validation_dataset = HousePriceDataset(dataset_path=validation_csv, dataset_type="train")

# Define input size based on data
input_size = train_dataset.input_dimension

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = NeuralNetHousePrice(input_size=input_size, hidden1=hidden_size1, hidden2=hidden_size2, hidden3=hidden_size3, output=output_size, dropout_prob=dropout_prob)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


min_val_loss = float('inf')
min_val_epoch = 0
best_model_state_dict = None

# Train for num_epochs epochs
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, labels in validation_loader:
            outputs = model(data)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

    # Check for early stopping
    if total_val_loss < min_val_loss:
        min_val_loss = total_val_loss
        min_val_epoch = epoch
        best_model_state_dict = copy.deepcopy(model.state_dict())
        patience_counter = 0
        
    print(f"Epoch #{epoch:3}, loss: {int(total_val_loss):5}, min loss: {int(min_val_loss):5} at epoch: {int(min_val_epoch):3}")


# Load best model state
if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)

# Testing
model.eval()
predictions = []
with torch.no_grad():
    for data in test_loader:
        outputs = model(data)
        predictions.extend(outputs.numpy().flatten())

os.chdir(os.path.join(os.getcwd(), "predictions"))

curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

# Save predictions to CSV
results_df = pd.DataFrame({'ID': range(1461, 1461 + len(predictions)), 'SalePrice': predictions})
results_df.to_csv(curr_time_str + 'predictions.csv', index=False)
