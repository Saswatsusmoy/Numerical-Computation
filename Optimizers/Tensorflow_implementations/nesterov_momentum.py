import tensorflow as tf

class NesterovMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Implements Nesterov Momentum optimization algorithm
        
        Args:
            learning_rate: float, learning rate
            momentum: float, momentum coefficient
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to variables using Nesterov Momentum
        
        Args:
            grads_and_vars: List of (gradient, variable) tuples
        """
        for grad, var in grads_and_vars:
            # Get variable name as unique key
            var_key = var.name
            
            # Initialize velocity if this is the first update for this variable
            if var_key not in self.velocities:
                self.velocities[var_key] = tf.zeros_like(var)
            
            # Get current velocity
            velocity = self.velocities[var_key]
            
            # Compute the Nesterov momentum update:
            # First look-ahead step
            look_ahead = var - self.momentum * velocity
            
            # Use the gradient at the look-ahead position
            # (Here we approximate by just using the current gradient)
            new_velocity = self.momentum * velocity - self.learning_rate * grad
            
            # Update variable
            var.assign(var + new_velocity)
            
            # Store the velocity for next iteration
            self.velocities[var_key] = new_velocity