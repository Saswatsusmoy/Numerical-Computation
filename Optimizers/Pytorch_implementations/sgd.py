import torch

class SGD(torch.optim.Optimizer):
    """
    Implements stochastic gradient descent (SGD) optimizer.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        
        # Initialize parent class (required for all PyTorch optimizers)
        super(SGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                    
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    
                    if 'momentum_buffer' not in param_state:
                        # Initialize momentum buffer
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Update parameters
                p.data.add_(d_p, alpha=-lr)
                
        return loss
