import torch
from torch.optim import Optimizer

class AdaDelta(Optimizer):
    """
    Implements AdaDelta algorithm.
    
    It has been proposed in "ADADELTA: An Adaptive Learning Rate Method".
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(rho=rho, eps=eps, weight_decay=weight_decay)
        super(AdaDelta, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)  # E[g²]
                    state['acc_delta'] = torch.zeros_like(p.data)   # E[Δx²]
                
                square_avg = state['square_avg']
                acc_delta = state['acc_delta']
                state['step'] += 1
                
                # Update running average of squared gradients
                square_avg.mul_(rho).addcmul_(grad, grad, value=1-rho)
                
                # Compute update
                std = torch.sqrt(square_avg.add(eps))
                delta = torch.sqrt(acc_delta.add(eps)).div(std).mul(grad)
                
                # Update parameters
                p.data.add_(-delta)
                
                # Update running average of squared updates
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1-rho)
                
        return loss