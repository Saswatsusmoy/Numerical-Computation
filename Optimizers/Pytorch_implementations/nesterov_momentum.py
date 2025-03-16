import torch
from torch.optim.optimizer import Optimizer

class NesterovMomentum(Optimizer):
    """Implements Nesterov Momentum algorithm.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, lr=0.001, momentum=0.9, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(NesterovMomentum, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(NesterovMomentum, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                if p.dim() > 1:
                    grad = grad.clone()
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                
                # Implement Nesterov momentum update
                # First, save the current parameter values
                old_p = torch.clone(p.data)
                
                # Look ahead with current momentum
                p.data.add_(buf, alpha=momentum)
                
                # Compute gradient at the lookahead point (already done by PyTorch autograd)
                
                # Update momentum buffer
                buf.mul_(momentum).add_(grad, alpha=-lr)
                
                # Reset parameters and apply the update
                p.data.copy_(old_p)
                p.data.add_(buf)
        
        return loss