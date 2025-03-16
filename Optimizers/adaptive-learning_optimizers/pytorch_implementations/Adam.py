import torch

class Adam:
    """
    Implements Adam optimizer from scratch.
    
    Arguments:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing 
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        self.params = list(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
    
    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    def step(self):
        """
        Performs a single optimization step.
        """
        for p in self.params:
            if p.grad is None:
                continue
                
            # Get gradients
            grad = p.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad.add(p.data, alpha=self.weight_decay)
            
            # State initialization
            if p not in self.state:
                self.state[p] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(p.data),  # First moment estimate
                    'exp_avg_sq': torch.zeros_like(p.data)  # Second moment estimate
                }
            
            state = self.state[p]
            
            # Increment step count
            state['step'] += 1
            
            # Update biased first moment estimate
            state['exp_avg'].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Update biased second raw moment estimate
            state['exp_avg_sq'].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Bias corrections
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            
            # Compute bias-corrected first moment estimate
            exp_avg_corrected = state['exp_avg'] / bias_correction1
            
            # Compute bias-corrected second raw moment estimate
            exp_avg_sq_corrected = state['exp_avg_sq'] / bias_correction2
            
            # Update parameters
            p.data.addcdiv_(exp_avg_corrected, exp_avg_sq_corrected.sqrt().add(self.eps), value=-self.lr)