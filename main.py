import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Parameters
BATCH = 128  # Batch size
EPOCHS = 10  # Training cycles
LR = 0.01  # Learning rate
TOP_RATIO = 0.1  # Fraction of gradients to keep
DATA_DIR = "./data"  # Directory to store the CIFAR-10 dataset

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize values
])

# Download CIFAR-10 dataset
print("Downloading CIFAR-10 dataset...")
train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
print("Download complete!")

# Create data loaders
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)


# Model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 2
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Fully connected 1
        self.fc2 = nn.Linear(256, 10)  # Fully connected 2
        self.pool = nn.MaxPool2d(2, 2)  # Pooling
        self.act = nn.ReLU()  # Activation

    def forward(self, x):
        x = self.act(self.conv1(x))  # Conv + activation
        x = self.pool(self.act(self.conv2(x)))  # Conv + activation + pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act(self.fc1(x))  # FC + activation
        return self.fc2(x)  # Output


# Gradient compression
def top_k_grad(grad, ratio):
    """
    Keep the top-k gradients by magnitude, set others to zero.

    Args:
        grad (Tensor): Gradient matrix.
        ratio (float): Fraction of gradients to keep.

    Returns:
        Tensor: Compressed gradients.
    """
    k = int(grad.numel() * ratio)  # Number of elements to retain
    if k == 0:
        return torch.zeros_like(grad)
    abs_grad = torch.abs(grad)
    _, idx = torch.topk(abs_grad.view(-1), k)  # Top-k indices
    compressed = torch.zeros_like(grad)  # Initialize compressed gradient
    compressed.view(-1)[idx] = grad.view(-1)[idx]  # Assign top-k values
    return compressed


# Training logic (using GPU)
def train(model, device, loader, optimizer, loss_fn, ratio):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for inputs, targets in loader:
            # Move inputs and targets to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Reset gradients
            preds = model(inputs)  # Forward pass
            loss = loss_fn(preds, targets)  # Compute loss
            loss.backward()  # Backward pass

            # Apply gradient compression
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = top_k_grad(param.grad, ratio)

            optimizer.step()  # Optimizer step
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(loader):.4f}")


# Evaluation logic (using GPU)
def evaluate(model, device, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            # Move inputs and targets to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)  # Forward pass
            total_loss += loss_fn(preds, targets).item()  # Accumulate loss
            correct += preds.argmax(dim=1).eq(targets).sum().item()  # Count correct predictions
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {correct / len(loader.dataset) * 100:.2f}%")


# Main function
def main():
    # Automatically detect and use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, move to GPU, and define loss function and optimizer
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Start training and evaluation
    train(model, device, train_loader, optimizer, loss_fn, TOP_RATIO)
    evaluate(model, device, test_loader, loss_fn)


# Entry point
if __name__ == "__main__":
    main()