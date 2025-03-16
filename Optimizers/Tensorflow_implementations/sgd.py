import tensorflow as tf

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, name="SGD"):
        """Initialize the SGD optimizer.
        
        Args:
            learning_rate: Learning rate (step size)
            momentum: Momentum coefficient (0.0 = no momentum)
            name: Optional name for the optimizer
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.name = name
        self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._velocity_dict = {}  # Store velocities for momentum
    
    def _get_velocity(self, var):
        """Get or create velocity variable for a given variable."""
        if self.momentum <= 0:
            return None
            
        # Use variable name as key
        name = var.name
        if name not in self._velocity_dict:
            self._velocity_dict[name] = tf.Variable(
                tf.zeros_like(var), 
                trainable=False,
                name=f"{self.name}_velocity_{name}"
            )
        return self._velocity_dict[name]
    
    def apply_gradients(self, grads_and_vars):
        """Apply gradients to variables.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs.
        
        Returns:
            The updated iteration count.
        """
        self.iterations.assign_add(1)
        
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            
            # Convert sparse gradients to dense if needed
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            
            if self.momentum > 0:
                velocity = self._get_velocity(var)
                
                # Update with momentum: v = momentum * v - learning_rate * grad
                new_velocity = self.momentum * velocity - self.learning_rate * grad
                velocity.assign(new_velocity)
                
                # Apply velocity to variable: w = w + v
                var.assign_add(new_velocity)
            else:
                # Standard SGD update: w = w - learning_rate * grad
                var.assign_sub(self.learning_rate * grad)
        
        return self.iterations