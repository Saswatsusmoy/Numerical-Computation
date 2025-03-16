import torch
from torch.optim import Optimizer

class AdaGrad(Optimizer):
    """
    Implements AdaGrad algorithm.
    
    AdaGrad adapts the learning rate of each parameter by dividing
    the learning rate by the square root of the sum of all the
    historical squared values of the gradient.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        eps (float, optional): term added to denominator for numerical stability (default: 1e-10)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(AdaGrad, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update sum of squared gradients
                state['sum'].addcmul_(grad, grad, value=1)
                
                # Update parameters
                std = state['sum'].sqrt().add(group['eps'])
                p.data.addcdiv_(grad, std, value=-group['lr'])
        
        return loss