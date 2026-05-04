"""
Main entry point - Assignment 2

Deep Learning Course, University of Deusto

Run this script to train your SimpleCNN on CIFAR-10 and generate
the visualizations required for your report.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WANDB_START_METHOD"] = "thread"

import wandb

wandb.init(
    project="dl-assignment2",
    entity="gonzalo-iglesias-deusto-u",

    config={
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "architecture": "SimpleCNN",
        "optimizer": "Adam",
        "regularization": "none",
    }
)

import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn import SimpleCNN
from train import get_data_loaders, train_model,log_sample_predictions,log_confusion_matrix


def plot_training_history(history, save_path='training_history.png'):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved training history to {save_path}")


def plot_confusion_matrix(model, test_loader, device,
                          save_path='confusion_matrix.png'):
    """Plot confusion matrix on the test set.

    Requires: pip install scikit-learn seaborn
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("Install scikit-learn and seaborn for confusion matrix:")
        print("  pip install scikit-learn seaborn")
        return

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved confusion matrix to {save_path}")


if __name__ == '__main__':
    # ---- Configuration ----
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    # ---- Data ----
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # ---- Model ----
    model = SimpleCNN(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCNN parameters: {num_params:,}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params})
    # ---- Train ----
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    history = train_model(model, train_loader, test_loader,
                          num_epochs=NUM_EPOCHS,
                          learning_rate=LEARNING_RATE)

    # ---- Results ----
    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.1f}%")
    print(f"Best test accuracy:  {max(history['test_acc']):.1f}%")

    # ---- Visualizations (for your report) ----
    plot_training_history(history)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_confusion_matrix(model, test_loader, device)
    CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


    log_confusion_matrix(model, test_loader, device, CIFAR10_CLASSES)
    log_sample_predictions(model, test_loader, device, CIFAR10_CLASSES)


    wandb.log({"training_history": wandb.Image('training_history.png')})
    wandb.log({"confusion_matrix_plot": wandb.Image('confusion_matrix.png')})

    wandb.finish()
