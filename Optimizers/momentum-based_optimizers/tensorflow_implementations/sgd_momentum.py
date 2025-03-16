import tensorflow as tf

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Stochastic Gradient Descent with Momentum optimizer implementation
        
        Args:
            learning_rate: Learning rate for gradient updates
            momentum: Momentum coefficient (between 0 and 1)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}  # Store velocity for each variable
    
    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to variables using momentum update rule
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs
            
        Returns:
            An operation that updates all variables
        """
        updates = []
        
        for grad, var in grads_and_vars:
            if grad is None:
                continue
                
            # Create velocity variable if it doesn't exist
            if var.ref() not in self.velocities:
                self.velocities[var.ref()] = tf.Variable(
                    tf.zeros_like(var), 
                    trainable=False,
                    name=f"{var.name.split(':')[0]}_velocity"
                )
            
            velocity = self.velocities[var.ref()]
            
            # Momentum update formula:
            # v = momentum * v - learning_rate * grad
            # var = var + v
            new_velocity = self.momentum * velocity - self.learning_rate * grad
            var_update = var.assign_add(new_velocity)
            velocity_update = velocity.assign(new_velocity)
            
            # Ensure the updates happen in the correct order
            with tf.control_dependencies([var_update]):
                updates.append(velocity_update)
        
        return tf.group(*updates) if updates else tf.no_op()
    
    def minimize(self, loss, var_list=None):
        """
        Minimize the loss by computing gradients and applying updates
        
        Args:
            loss: Loss tensor to minimize
            var_list: List of variables to update (defaults to all trainable variables)
            
        Returns:
            An operation that minimizes the loss
        """
        if var_list is None:
            var_list = tf.compat.v1.trainable_variables()
        
        grads = tf.gradients(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        
        return self.apply_gradients(grads_and_vars)