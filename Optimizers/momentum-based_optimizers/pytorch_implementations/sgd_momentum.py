import torch

class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0):
        """
        Implementation of SGD with momentum optimizer
        
        Args:
            parameters: An iterable of parameters to optimize
            lr: Learning rate (default: 0.01)
            momentum: Momentum factor (default: 0.9)
            weight_decay: Weight decay factor (L2 penalty) (default: 0)
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize velocity for each parameter
        self.velocity = [torch.zeros_like(param) for param in self.parameters]
        
    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
                
    def step(self):
        """
        Performs a single optimization step.
        """
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay if specified
            if self.weight_decay > 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
                
            # Update velocity using momentum
            self.velocity[i].mul_(self.momentum).add_(grad)
            
            # Update parameters using velocity
            param.data.add_(self.velocity[i], alpha=-self.lr)