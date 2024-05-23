# Import required libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Configure the device for computation (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
input_size = 784  # Input size for images (28x28)
hidden_size = 100  # Number of neurons in the hidden layer
num_classes = 10  # Number of classes (digits 0 to 9)
num_epochs = 2  # Number of training epochs
batch_size = 100  # Batch size for training
learning_rate = 0.001  # Learning rate for the optimizer

# Load the MNIST dataset for training and testing
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create an iterator for the training data loader
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

# Display the first 6 sample images
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')

# Define a feedforward neural network class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # First linear layer
        # The first linear layer (self.l1) maps the input image features (flattened)
        #  to an intermediate representation in the hidden layer 
        # with non-linearities introduced through activation functions.
        self.relu = nn.ReLU()  # ReLU activation function
        self.l2 = nn.Linear(hidden_size, num_classes)  # Second linear layer
        # The second linear layer (self.l2) takes the output of the first linear layer 
        # and produces the final output, typically used for classification tasks, 
        # representing the model's predictions for different classes (e.g., digits from 0 to 9 in this case).

    def forward(self, x):
        out = self.l1(x)  # Forward pass through the first linear layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.l2(out)  # Forward pass through the second linear layer
        return out

# Create an instance of the neural network model
model = NeuralNet(input_size, hidden_size, num_classes)

# Define the loss function (cross-entropy) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
# The cross-entropy loss function, often used in machine learning,
#  measures the dissimilarity between predicted probabilities and actual class labels. 
# It quantifies how well a model's predicted probability distribution
#  matches the true distribution of class labels, 
# penalizing deviations from the correct classification.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# The Adam optimizer is a popular optimization algorithm for training deep neural networks 
# that combines adaptive learning rates, momentum, and bias correction
#  to efficiently update model parameters during training. 
# It dynamically adjusts learning rates for individual parameters, 
# helping models converge quickly and effectively.

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Prepare the data for forward pass
        images = images.reshape(-1, 28 * 28).to(device)  # Reshape images and move to the specified device
        labels = labels.to(device)  # Move labels to the specified device

        # Forward pass
        outputs = model(images)  # Forward pass through the neural network
        # calls forward function
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the model parameters

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# Testing the trained model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # Get the predicted labels
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples  # Calculate accuracy in percentage
    print(f'accuracy = {acc}')
