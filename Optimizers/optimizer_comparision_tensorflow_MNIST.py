import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import custom optimizers
from Tensorflow_implementations.sgd import SGD
from Tensorflow_implementations.sgd_momentum import SGDMomentum
from Tensorflow_implementations.nesterov_momentum import NesterovMomentum

# Check for GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Define MNIST Neural Network using Keras
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Data loading function
def load_data(batch_size=64):
    # Load MNIST dataset from TensorFlow datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape the data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset, x_test, y_test

# Custom training loop for TensorFlow
def train(model, train_dataset, optimizer, loss_fn, metrics):
    train_loss = 0
    correct = 0
    total = 0
    batch_times = []
    
    start_time = time.time()
    
    # Define training step function
    @tf.function
    def train_step(x, y, optimizer_name):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        
        # Get gradients
        grads = tape.gradient(loss_value, model.trainable_variables)
        
        # Apply gradients based on optimizer type
        if optimizer_name == "SGD":
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        elif optimizer_name == "SGD Momentum" or optimizer_name == "Nesterov Momentum":
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        else:  # TensorFlow built-in optimizer
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        return loss_value, logits
    
    # Get optimizer name for dispatch
    if isinstance(optimizer, SGD):
        optimizer_name = "SGD"
    elif isinstance(optimizer, SGDMomentum):
        optimizer_name = "SGD Momentum"
    elif isinstance(optimizer, NesterovMomentum):
        optimizer_name = "Nesterov Momentum"
    else:
        optimizer_name = "TF Optimizer"
    
    batch_idx = 0
    for x_batch, y_batch in train_dataset:
        batch_start = time.time()
        
        # Train step
        loss_value, logits = train_step(x_batch, y_batch, optimizer_name)
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        # Calculate metrics
        train_loss += loss_value.numpy()
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32)).numpy()
        total += len(y_batch)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss: {loss_value.numpy():.6f}')
        batch_idx += 1
    
    epoch_time = time.time() - start_time
    train_loss /= batch_idx
    train_accuracy = 100. * correct / total
    
    # Store metrics
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_accuracy)
    metrics['train_time'].append(epoch_time)
    metrics['batch_times'].extend(batch_times)
    
    print(f'Train set: Average loss: {train_loss:.4f}, '
          f'Accuracy: {int(correct)}/{total} ({train_accuracy:.2f}%), '
          f'Time: {epoch_time:.2f}s')
    
    return metrics

# Testing function
def test(model, test_dataset, loss_fn, metrics, x_test=None, y_test=None):
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # Define test step function
    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        loss_value = loss_fn(y, logits)
        return loss_value, logits
    
    batch_idx = 0
    for x_batch, y_batch in test_dataset:
        # Test step
        loss_value, logits = test_step(x_batch, y_batch)
        
        # Calculate metrics
        test_loss += loss_value.numpy()
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32)).numpy()
        total += len(y_batch)
        
        # Store for confusion matrix
        all_preds.extend(predictions.numpy())
        all_targets.extend(labels.numpy())
        
        batch_idx += 1
    
    test_loss /= batch_idx
    test_accuracy = 100. * correct / total
    
    # Store metrics
    metrics['test_loss'].append(test_loss)
    metrics['test_acc'].append(test_accuracy)
    metrics['confusion_matrix'] = confusion_matrix(all_targets, all_preds)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {int(correct)}/{total} ({test_accuracy:.2f}%)')
    
    return metrics

# Calculate parameter updates
def calculate_parameter_updates(model, optimizer, train_dataset, steps=10):
    param_updates = [tf.zeros_like(var) for var in model.trainable_variables]
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Define update step function
    @tf.function
    def update_step(x, y, optimizer_name):
        # Store old parameters
        old_params = [tf.identity(var) for var in model.trainable_variables]
        
        # Forward and backward pass
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        
        # Get gradients
        grads = tape.gradient(loss_value, model.trainable_variables)
        
        # Apply gradients based on optimizer type
        if optimizer_name == "SGD":
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        elif optimizer_name == "SGD Momentum" or optimizer_name == "Nesterov Momentum":
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        else:  # TensorFlow built-in optimizer
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Calculate updates
        updates = [tf.abs(new - old) for new, old in zip(model.trainable_variables, old_params)]
        return updates
    
    # Get optimizer name for dispatch
    if isinstance(optimizer, SGD):
        optimizer_name = "SGD"
    elif isinstance(optimizer, SGDMomentum):
        optimizer_name = "SGD Momentum"
    elif isinstance(optimizer, NesterovMomentum):
        optimizer_name = "Nesterov Momentum"
    else:
        optimizer_name = "TF Optimizer"
    
    # Run updates for specified number of steps
    step_count = 0
    for x_batch, y_batch in train_dataset:
        if step_count >= steps:
            break
            
        # Get updates for this batch
        batch_updates = update_step(x_batch, y_batch, optimizer_name)
        
        # Accumulate updates
        for i, update in enumerate(batch_updates):
            param_updates[i] += update
            
        step_count += 1
    
    # Average updates
    for i in range(len(param_updates)):
        param_updates[i] = param_updates[i] / steps
    
    # Calculate mean update magnitude across all parameters
    update_magnitude = tf.reduce_mean([tf.reduce_mean(update) for update in param_updates]).numpy()
    
    return update_magnitude

# Main training function
def run_experiment(optimizer_name, optimizer, epochs, batch_size):
    print(f"\nRunning experiment with {optimizer_name}")
    train_dataset, test_dataset, x_test, y_test = load_data(batch_size)
    
    model = create_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    metrics = defaultdict(list)
    
    # Track initial parameter update magnitude
    initial_update = calculate_parameter_updates(model, optimizer, train_dataset)
    metrics['param_update'] = [initial_update]
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        metrics = train(model, train_dataset, optimizer, loss_fn, metrics)
        metrics = test(model, test_dataset, loss_fn, metrics, x_test, y_test)
        
        # Track parameter update magnitude every epoch
        if epoch < epochs:  # Skip last epoch as we won't use its updates
            update_magnitude = calculate_parameter_updates(model, optimizer, train_dataset)
            metrics['param_update'].append(update_magnitude)
    
    return metrics

# Plot comparison metrics (same as PyTorch version)
def plot_comparison(all_metrics, optimizer_names):
    # Plot training and test loss
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, name in enumerate(optimizer_names):
        axes[0, 0].plot(all_metrics[name]['train_loss'], label=name)
        axes[0, 1].plot(all_metrics[name]['test_loss'], label=name)
        axes[1, 0].plot(all_metrics[name]['train_acc'], label=name)
        axes[1, 1].plot(all_metrics[name]['test_acc'], label=name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('tensorflow_optimizer_comparison_learning_curves.png')
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    avg_times = [np.mean(all_metrics[name]['train_time']) for name in optimizer_names]
    plt.bar(optimizer_names, avg_times)
    plt.title('Average Training Time per Epoch')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tensorflow_optimizer_comparison_time.png')
    
    # Plot parameter update magnitudes
    plt.figure(figsize=(10, 6))
    for name in optimizer_names:
        plt.plot(all_metrics[name]['param_update'], label=name)
    plt.title('Parameter Update Magnitude')
    plt.xlabel('Epochs')
    plt.ylabel('Average Absolute Update')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tensorflow_optimizer_comparison_param_updates.png')
    
    # Create a table of final metrics
    final_metrics = {
        'Optimizer': [],
        'Final Train Acc (%)': [],
        'Final Test Acc (%)': [],
        'Avg Time per Epoch (s)': [],
        'Generalization Gap (%)': [],
        'Convergence Speed': []
    }
    
    for name in optimizer_names:
        metrics = all_metrics[name]
        
        # Calculate convergence speed (epochs to reach 90% of final test accuracy)
        final_acc = metrics['test_acc'][-1]
        threshold = 0.9 * final_acc
        convergence_epoch = next((i + 1 for i, acc in enumerate(metrics['test_acc']) 
                                if acc >= threshold), len(metrics['test_acc']))
        
        final_metrics['Optimizer'].append(name)
        final_metrics['Final Train Acc (%)'].append(f"{metrics['train_acc'][-1]:.2f}")
        final_metrics['Final Test Acc (%)'].append(f"{metrics['test_acc'][-1]:.2f}")
        final_metrics['Avg Time per Epoch (s)'].append(f"{np.mean(metrics['train_time']):.2f}")
        final_metrics['Generalization Gap (%)'].append(f"{metrics['train_acc'][-1] - metrics['test_acc'][-1]:.2f}")
        final_metrics['Convergence Speed'].append(f"{convergence_epoch} epochs")
    
    # Print the table
    print("\n--- Optimizer Comparison Summary ---")
    df = pd.DataFrame(final_metrics)
    print(df.to_string(index=False))
    
    # Save metrics table to CSV
    df.to_csv('tensorflow_optimizer_comparison_metrics.csv', index=False)

def main():
    # Hyperparameters
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    
    # Create model
    model = create_model()
    
    # Initialize optimizers
    custom_sgd = SGD(learning_rate=learning_rate)
    custom_sgd_momentum = SGDMomentum(learning_rate=learning_rate, momentum=momentum)
    custom_nesterov = NesterovMomentum(learning_rate=learning_rate, momentum=momentum)
    tf_sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Run experiments
    optimizers = {
        'Custom SGD': custom_sgd,
        'Custom SGD Momentum': custom_sgd_momentum,
        'Custom Nesterov': custom_nesterov,
        'TensorFlow SGD': tf_sgd
    }
    
    all_metrics = {}
    
    for name, opt in optimizers.items():
        # Create a fresh model for each optimizer
        tf.keras.backend.clear_session()
        model = create_model()
        
        # Run experiment with this optimizer
        all_metrics[name] = run_experiment(name, opt, epochs, batch_size)
    
    # Plot and save comparison metrics
    plot_comparison(all_metrics, list(optimizers.keys()))

if __name__ == '__main__':
    main()
