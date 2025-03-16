import torch

class RMSProp(torch.optim.Optimizer):
    """
    Implements RMSProp algorithm.
    
    The algorithm adapts the learning rate for each parameter based on the 
    history of squared gradients.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        centered (bool, optional): if True, compute the centered RMSProp, using gradients normalized
            by the variance of the incoming gradients (default: False)
    """
    
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, centered=centered)
        super(RMSProp, self).__init__(params, defaults)
        
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
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSProp does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update running average of squared gradients
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                
                p.data.addcdiv_(grad, avg, value=-group['lr'])
                
        return loss