import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import custom optimizers
from Pytorch_implementations.sgd import SGD
from Pytorch_implementations.sgd_momentum import SGDMomentum
from Pytorch_implementations.nesterov_momentum import NesterovMomentum

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define MNIST Neural Network
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Data loading function
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, metrics):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_times = []
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        data, target = data.to(device), target.to(device)
        
        # Handle different optimizer interfaces
        if isinstance(optimizer, SGDMomentum):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    epoch_time = time.time() - start_time
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / total
    
    # Store metrics
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_accuracy)
    metrics['train_time'].append(epoch_time)
    metrics['batch_times'].extend(batch_times)
    
    print(f'Epoch {epoch}: Train set: Average loss: {train_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({train_accuracy:.2f}%), '
          f'Time: {epoch_time:.2f}s')
    
    return metrics

# Testing function
def test(model, device, test_loader, criterion, metrics):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    # Store metrics
    metrics['test_loss'].append(test_loss)
    metrics['test_acc'].append(test_accuracy)
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_preds)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_accuracy:.2f}%)')
    
    return metrics

# Calculate parameter updates
def calculate_parameter_updates(model, optimizer, train_loader, device, steps=10):
    model.train()
    param_updates = []
    
    for param in model.parameters():
        param_updates.append(torch.zeros_like(param))
    
    criterion = nn.CrossEntropyLoss()
    
    for i, (data, target) in enumerate(train_loader):
        if i >= steps:
            break
        
        data, target = data.to(device), target.to(device)
        
        # Store old parameters
        old_params = [p.clone().detach() for p in model.parameters()]
        
        # Forward pass and backward pass
        if isinstance(optimizer, SGDMomentum):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Calculate updates
        for i, (old, new) in enumerate(zip(old_params, model.parameters())):
            param_updates[i] += (new - old).abs()
    
    # Average updates
    for i in range(len(param_updates)):
        param_updates[i] /= steps
    
    return torch.cat([p.flatten() for p in param_updates]).mean().item()

# Main training function
def run_experiment(optimizer_name, optimizer, epochs, batch_size):
    print(f"\nRunning experiment with {optimizer_name}")
    train_loader, test_loader = load_data(batch_size)
    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    metrics = defaultdict(list)
    
    # Track initial parameter update magnitude
    initial_update = calculate_parameter_updates(model, optimizer, train_loader, device)
    metrics['param_update'] = [initial_update]
    
    for epoch in range(1, epochs + 1):
        metrics = train(model, device, train_loader, optimizer, criterion, epoch, metrics)
        metrics = test(model, device, test_loader, criterion, metrics)
        
        # Track parameter update magnitude every epoch
        if epoch < epochs:  # Skip last epoch as we won't use its updates
            update_magnitude = calculate_parameter_updates(model, optimizer, train_loader, device)
            metrics['param_update'].append(update_magnitude)
    
    return metrics

# Plot comparison metrics
def plot_comparison(all_metrics, optimizer_names):
    # Plot training and test loss
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, name in enumerate(optimizer_names):
        axes[0, 0].plot(all_metrics[name]['train_loss'], label=name)
        axes[0, 1].plot(all_metrics[name]['test_loss'], label=name)
        axes[1, 0].plot(all_metrics[name]['train_acc'], label=name)
        axes[1, 1].plot(all_metrics[name]['test_acc'], label=name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_learning_curves.png')
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    avg_times = [np.mean(all_metrics[name]['train_time']) for name in optimizer_names]
    plt.bar(optimizer_names, avg_times)
    plt.title('Average Training Time per Epoch')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('optimizer_comparison_time.png')
    
    # Plot parameter update magnitudes
    plt.figure(figsize=(10, 6))
    for name in optimizer_names:
        plt.plot(all_metrics[name]['param_update'], label=name)
    plt.title('Parameter Update Magnitude')
    plt.xlabel('Epochs')
    plt.ylabel('Average Absolute Update')
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimizer_comparison_param_updates.png')
    
    # Create a table of final metrics
    final_metrics = {
        'Optimizer': [],
        'Final Train Acc (%)': [],
        'Final Test Acc (%)': [],
        'Avg Time per Epoch (s)': [],
        'Generalization Gap (%)': [],
        'Convergence Speed': []
    }
    
    for name in optimizer_names:
        metrics = all_metrics[name]
        
        # Calculate convergence speed (epochs to reach 90% of final test accuracy)
        final_acc = metrics['test_acc'][-1]
        threshold = 0.9 * final_acc
        convergence_epoch = next((i + 1 for i, acc in enumerate(metrics['test_acc']) 
                                if acc >= threshold), len(metrics['test_acc']))
        
        final_metrics['Optimizer'].append(name)
        final_metrics['Final Train Acc (%)'].append(f"{metrics['train_acc'][-1]:.2f}")
        final_metrics['Final Test Acc (%)'].append(f"{metrics['test_acc'][-1]:.2f}")
        final_metrics['Avg Time per Epoch (s)'].append(f"{np.mean(metrics['train_time']):.2f}")
        final_metrics['Generalization Gap (%)'].append(f"{metrics['train_acc'][-1] - metrics['test_acc'][-1]:.2f}")
        final_metrics['Convergence Speed'].append(f"{convergence_epoch} epochs")
    
    # Print the table
    print("\n--- Optimizer Comparison Summary ---")
    df = pd.DataFrame(final_metrics)
    print(df.to_string(index=False))
    
    # Save metrics table to CSV
    df.to_csv('optimizer_comparison_metrics.csv', index=False)

def main():
    # Hyperparameters
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    
    # Initialize models and optimizers
    model_sgd = MNISTNet().to(device)
    model_sgd_momentum = MNISTNet().to(device)
    model_nesterov = MNISTNet().to(device)
    model_pytorch_sgd = MNISTNet().to(device)
    
    # Initialize optimizers
    optimizer_sgd = SGD(model_sgd.parameters(), lr=learning_rate)
    optimizer_sgd_momentum = SGDMomentum(model_sgd_momentum.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_nesterov = NesterovMomentum(model_nesterov.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_pytorch_sgd = torch.optim.SGD(model_pytorch_sgd.parameters(), lr=learning_rate)
    
    # Run experiments
    optimizers = {
        'Custom SGD': optimizer_sgd,
        'Custom SGD Momentum': optimizer_sgd_momentum,
        'Custom Nesterov': optimizer_nesterov,
        'PyTorch SGD': optimizer_pytorch_sgd
    }
    
    all_metrics = {}
    
    for name, opt in optimizers.items():
        model = MNISTNet().to(device)
        if name == 'Custom SGD Momentum':
            opt = SGDMomentum(model.parameters(), lr=learning_rate, momentum=momentum)
        elif name == 'Custom Nesterov':
            opt = NesterovMomentum(model.parameters(), lr=learning_rate, momentum=momentum)
        elif name == 'Custom SGD':
            opt = SGD(model.parameters(), lr=learning_rate)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
        all_metrics[name] = run_experiment(name, opt, epochs, batch_size)
    
    # Plot and save comparison metrics
    plot_comparison(all_metrics, list(optimizers.keys()))

if __name__ == '__main__':
    main()
