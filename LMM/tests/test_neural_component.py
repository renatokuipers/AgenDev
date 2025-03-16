#!/usr/bin/env python
# test_neural_component.py - Tests for neural components

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LMM.core.base.component import NeuralComponent, ComponentConfig, DevelopmentalStage

class TestFeedForwardComponent(NeuralComponent):
    """Test implementation of a neural component with a feedforward network."""
    
    def _build_network(self) -> None:
        """Build the feedforward neural network architecture"""
        layers = []
        
        # Input layer
        in_features = self.config.input_dim
        
        # Add hidden layers based on configuration
        if hasattr(self.config, 'hidden_dims') and self.config.hidden_dims:
            for hidden_dim in self.config.hidden_dims:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(self.get_activation_function())
                if hasattr(self.config, 'dropout_rate') and self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
                in_features = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_features, self.config.output_dim))
        
        # For classification, add sigmoid for binary or softmax for multi-class
        if hasattr(self.config, 'task') and self.config.task == 'classification':
            if self.config.output_dim == 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softmax(dim=1))
        
        # Create sequential model
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the component"""
        return self.net(x)
    
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the component to a developmental stage.
        
        This method demonstrates how a component can adapt its behavior based on
        developmental stages, mimicking cognitive development:
        
        - SENSORIMOTOR: Very basic learning with high regularization
        - PREOPERATIONAL: More capacity but still high regularization
        - CONCRETE_OPERATIONAL: Balanced capacity and regularization
        - FORMAL_OPERATIONAL: Full capacity with minimal regularization
        """
        # Get all dropout layers
        dropout_layers = [m for m in self.net.modules() if isinstance(m, nn.Dropout)]
        
        # Adjust dropout rate based on developmental stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            # High regularization in early stages
            dropout_rate = 0.5
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            # Slightly lower regularization
            dropout_rate = 0.3
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            # Balanced regularization
            dropout_rate = 0.2
        else:  # FORMAL_OPERATIONAL
            # Minimal regularization
            dropout_rate = 0.1
        
        # Update dropout rates
        for layer in dropout_layers:
            layer.p = dropout_rate
        
        if self.debug_mode:
            print(f"Adapted to stage {stage.value} with dropout rate {dropout_rate}")


def generate_spiral_data(samples_per_class=100, noise=0.1, seed=42):
    """Generate a spiral dataset for classification"""
    np.random.seed(seed)
    
    angles = np.linspace(0, 2*np.pi, samples_per_class)
    
    # Generate data for class 0
    radius0 = np.linspace(0, 1, samples_per_class)
    x0 = radius0 * np.cos(angles)
    y0 = radius0 * np.sin(angles)
    
    # Generate data for class 1
    radius1 = np.linspace(0, 1, samples_per_class)
    x1 = radius1 * np.cos(angles + np.pi)
    y1 = radius1 * np.sin(angles + np.pi)
    
    # Add noise
    x0 += np.random.normal(0, noise, samples_per_class)
    y0 += np.random.normal(0, noise, samples_per_class)
    x1 += np.random.normal(0, noise, samples_per_class)
    y1 += np.random.normal(0, noise, samples_per_class)
    
    # Combine data
    X = np.vstack([np.column_stack((x0, y0)), np.column_stack((x1, y1))])
    y = np.hstack([np.zeros(samples_per_class), np.ones(samples_per_class)])
    
    return X, y


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary of the model"""
    h = 0.02  # step size in the mesh
    
    # Create a mesh grid for plotting the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Convert to PyTorch tensors and predict
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy()
        Z = (Z > 0.5).astype(int).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot the data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', edgecolors='k', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='^', edgecolors='k', label='Class 1')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def train_developmental_stages(component, X_train, y_train, X_test, y_test, 
                              epochs_per_stage=50, learning_rate=0.01, batch_size=32):
    """Train a component through all developmental stages"""
    # Training settings
    criterion = nn.BCELoss()
    
    # Tracking metrics
    train_losses = []
    test_accuracies = []
    decision_boundaries = []
    
    # Train through each developmental stage
    stages = [
        DevelopmentalStage.SENSORIMOTOR,
        DevelopmentalStage.PREOPERATIONAL,
        DevelopmentalStage.CONCRETE_OPERATIONAL,
        DevelopmentalStage.FORMAL_OPERATIONAL
    ]
    
    # Create PyTorch datasets
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    for stage in stages:
        print(f"\n--- Training in {stage.value} stage ---")
        
        # Adapt component to the current developmental stage
        component.adapt_to_developmental_stage(stage)
        
        # Create optimizer (new for each stage as parameters might change)
        optimizer = optim.Adam(component.parameters(), lr=learning_rate)
        
        # Training loop for this stage
        stage_losses = []
        
        for epoch in range(epochs_per_stage):
            # Training mode
            component.train()
            
            # Shuffle data for each epoch
            indices = torch.randperm(len(X_train_tensor))
            
            # Mini-batch training
            for start_idx in range(0, len(X_train_tensor), batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = component(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Record loss
                stage_losses.append(loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                # Evaluation mode
                component.eval()
                
                with torch.no_grad():
                    # Calculate accuracy on test set
                    test_outputs = component(X_test_tensor)
                    test_preds = (test_outputs > 0.5).float()
                    accuracy = (test_preds == y_test_tensor).float().mean().item()
                    
                    print(f"Epoch {epoch+1}/{epochs_per_stage}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Store final stage metrics
        train_losses.extend(stage_losses)
        
        # Evaluate after stage completion
        component.eval()
        with torch.no_grad():
            # Calculate accuracy on test set
            test_outputs = component(X_test_tensor)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test_tensor).float().mean().item()
            test_accuracies.append(accuracy)
            
            # Save decision boundary plot for this stage
            db_plot = plot_decision_boundary(component, X_test, y_test, 
                                           title=f"Decision Boundary: {stage.value} Stage")
            decision_boundaries.append(db_plot)
        
        print(f"Completed {stage.value} stage with test accuracy: {accuracy:.4f}")
    
    # Plot training loss across all stages
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Across Developmental Stages')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.axvline(x=epochs_per_stage, color='r', linestyle='--', label='Stage Change')
    plt.axvline(x=epochs_per_stage*2, color='r', linestyle='--')
    plt.axvline(x=epochs_per_stage*3, color='r', linestyle='--')
    plt.legend(['Loss', 'Stage Change'])
    plt.tight_layout()
    
    # Print final metrics for each stage
    for i, stage in enumerate(stages):
        print(f"{stage.value} stage final test accuracy: {test_accuracies[i]:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'decision_boundaries': decision_boundaries
    }


def run_test():
    """Run a complete test of the neural component through developmental stages"""
    # Generate spiral dataset
    print("Generating spiral dataset...")
    X, y = generate_spiral_data(samples_per_class=200, noise=0.15)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the neural component
    print("Creating neural component...")
    component_config = ComponentConfig(
        component_id="test_component",
        component_type="feedforward",
        name="Test Feedforward Component",
        description="A test component for developmental learning",
        input_dim=2,  # 2D spiral data
        output_dim=1,  # Binary classification
        hidden_dims=[20, 10],  # Hidden layers
        activation="relu",
        dropout_rate=0.2,
        task="classification",  # Specify classification task
        initial_stage=DevelopmentalStage.SENSORIMOTOR  # Start from earliest stage
    )
    
    component = TestFeedForwardComponent(component_config)
    component.debug_mode = True  # Enable debug mode
    
    # Train through developmental stages
    print("Training through developmental stages...")
    results = train_developmental_stages(
        component, X_train, y_train, X_test, y_test,
        epochs_per_stage=50,  # 50 epochs per stage
        learning_rate=0.01,
        batch_size=32
    )
    
    print("\nTraining complete!")
    
    # Save the final model
    save_dir = "tmp"
    os.makedirs(save_dir, exist_ok=True)
    saved_path = component.save(save_dir)
    print(f"Model saved to: {saved_path}")
    
    # Show the final decision boundary
    plt.show()


if __name__ == "__main__":
    run_test() 