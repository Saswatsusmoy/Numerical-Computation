import torch
import numpy as np

class MBGradientDescent:
    """
    Mini-batch Gradient Descent optimizer implemented from scratch using PyTorch
    
    Args:
        params: An iterable of parameters to optimize
        lr: Learning rate (step size)
    """
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def zero_grad(self):
        """Reset gradients to zero for all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Perform one optimization step"""
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad

def train_with_mb_gradient_descent(model, loss_fn, X, y, batch_size=32, lr=0.01, epochs=100):
    """
    Train a model using mini-batch gradient descent
    
    Args:
        model: PyTorch model
        loss_fn: Loss function
        X: Input tensor
        y: Target tensor
        batch_size: Size of mini-batches
        lr: Learning rate
        epochs: Number of training epochs
    """
    optimizer = MBGradientDescent(model.parameters(), lr=lr)
    n_samples = len(X)
    
    for epoch in range(epochs):
        # Shuffle data at the beginning of each epoch
        indices = torch.randperm(n_samples)
        running_loss = 0.0
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            # Get mini-batch indices
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            
            # Extract batch data
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/((n_samples-1)//batch_size + 1):.6f}")