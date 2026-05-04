"""
Training Pipeline for CIFAR-10

Assignment 2 - Deep Learning Course, University of Deusto

Implement data loading, training, and evaluation functions for
training your SimpleCNN on the CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix



def get_data_loaders(batch_size=128, data_dir='./data'):
    """
    Load the CIFAR-10 dataset and create DataLoaders.

    Requirements:
        - Training set: apply RandomHorizontalFlip, ToTensor, and Normalize
          with CIFAR-10 statistics: mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010)
        - Test set: apply only ToTensor and Normalize (no augmentation)
        - Return DataLoaders with the given batch_size
        - Training loader should shuffle, test loader should not

    Args:
        batch_size: Batch size for DataLoaders (default: 128)
        data_dir:   Directory to download/store CIFAR-10 (default: './data')

    Returns:
        train_loader: DataLoader for training set
        test_loader:  DataLoader for test set
    """
    # TODO: Define transforms, load datasets, create DataLoaders
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),         
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([           
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Requirements:
        - Set model to training mode
        - For each batch: zero gradients, forward pass, compute loss,
          backward pass, optimizer step
        - Track and return average loss and accuracy for the epoch

    Args:
        model:        The CNN model
        train_loader: DataLoader for training data
        criterion:    Loss function (e.g., CrossEntropyLoss)
        optimizer:    Optimizer (e.g., Adam)
        device:       Device to use ('cuda' or 'cpu')

    Returns:
        avg_loss:  Average training loss over the epoch
        accuracy:  Training accuracy as a percentage (0-100)
    """
    # TODO: Implement training for one epoch
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)  
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = out.max(1)       
        
        total += labels.size(0)        
        correct += predicted.eq(labels).sum().item()  
            
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
            
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Requirements:
        - Set model to eval mode
        - Use torch.no_grad() context
        - Compute and return test accuracy as a percentage

    Args:
        model:       The CNN model
        test_loader: DataLoader for test data
        device:      Device to use ('cuda' or 'cpu')

    Returns:
        accuracy: Test accuracy as a percentage (0-100)
    """
    # TODO: Implement evaluation
    model.eval()
    correct=0
    total = 0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = outputs.max(1)
            total+=labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
            
    

def train_model(model, train_loader, test_loader,
                num_epochs=10, learning_rate=0.001):
    """
    Full training pipeline.

    Requirements:
        - Use CrossEntropyLoss and Adam optimizer
        - Train for num_epochs, tracking loss and accuracy each epoch
        - Evaluate on test set after each epoch
        - Print progress each epoch
        - Return training history

    Args:
        model:         The CNN model
        train_loader:  DataLoader for training data
        test_loader:   DataLoader for test data
        num_epochs:    Number of training epochs (default: 10)
        learning_rate: Learning rate for Adam (default: 0.001)

    Returns:
        history: dict with keys:
            'train_loss': list of average losses per epoch
            'train_acc':  list of training accuracies per epoch
            'test_acc':   list of test accuracies per epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")
        # W&B logging por epoch
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

    return history
        


def log_confusion_matrix(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })
    
    

def log_sample_predictions(model, test_loader, device, class_names, num_images=5):
    model.eval()
    images_logged = []
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    for i in range(num_images):
        img = images[i].cpu()
        # Desnormalizar para que se vea bien
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        images_logged.append(wandb.Image(img, caption=(
            f"GT: {class_names[labels[i]]} | "
            f"Pred: {class_names[predicted[i]]}"
        )))

    wandb.log({"sample_predictions": images_logged})
    


