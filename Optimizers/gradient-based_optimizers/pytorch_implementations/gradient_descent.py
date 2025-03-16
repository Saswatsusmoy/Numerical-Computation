import torch

class GradientDescent:
    """
    Implements basic gradient descent optimization algorithm.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 0.01)
    """
    
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        self.params = list(params)
        self.lr = lr
        
    def step(self):
        """
        Performs a single optimization step.
        
        Updates each parameter based on its gradient:
        param = param - lr * grad
        """
        for param in self.params:
            if param.grad is not None:
                param.data.add_(param.grad, alpha=-self.lr)
                
    def zero_grad(self):
        """
        Zeros out all the gradients of the parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()