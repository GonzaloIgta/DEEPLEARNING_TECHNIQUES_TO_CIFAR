"""
Part 2: Transfer Learning on CIFAR-100

Assignment 2 - Deep Learning Course, University of Deusto

Compare a from-scratch CNN against a fine-tuned pretrained ResNet-18
on CIFAR-100 (100 classes).  All results are logged to W&B.

Usage:
    python transfer_learning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import wandb
from experiments import ConfigurableCNN, log_prediction_table

# ── Constants ────────────────────────────────────────────────────────
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_NUM_CLASSES = 100

WANDB_PROJECT = "dl-assignment2"      

# ── Data ─────────────────────────────────────────────────────────────

def get_cifar100_loaders(batch_size=128, data_dir='./data'):
    """
    Load CIFAR-100 with standard transforms.

    TODO: This is almost identical to your CIFAR-10 loader.
          Use torchvision.datasets.CIFAR100 instead of CIFAR10
          and the CIFAR-100 mean/std above.
    """
    # TODO: Define train and test transforms (same structure as CIFAR-10)
    # TODO: Load CIFAR-100 datasets
    # TODO: Create and return DataLoaders
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),         
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN,
                             CIFAR100_STD)
    ])
    test_transform = transforms.Compose([           
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN,
                             CIFAR100_STD)
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 


# ── From-Scratch CNN for CIFAR-100 ───────────────────────────────────

class CNN100(ConfigurableCNN):
    """
    A CNN for CIFAR-100 (from scratch).

    Use the same architecture as your best model from experiments.py,
    but change the output to 100 classes.

    Hint: You can copy your ConfigurableCNN and set num_classes=100,
    or define a fixed architecture here.
    """
    exp = {"Depper_CN100": {
        "num_blocks": 3, "filters_start": 16, "kernel_size": 3,
        "use_batchnorm": False, "dropout_rate": 0.0,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "grad_clip": 0, "batch_size": 128, "epochs": 15, 
        "num_classes": 100,
    }}
    def __init__(self, num_classes = 100 ):
        exp = {**self.exp["Depper_CN100"], "num_classes": num_classes}  
        super().__init__(exp)

       
   

# ── ResNet-18 for Transfer Learning ──────────────────────────────────

def create_resnet18(num_classes=100):
    """
    Load a pretrained ResNet-18 and replace the final FC layer.

    TODO:
        1. Load ResNet-18 with pretrained ImageNet weights.
        2. Replace model.fc with a new nn.Linear that outputs
           `num_classes` classes.
        3. Return the model.

    Hint:
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    """
    # TODO: Implement this function
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model



# ── Training helpers ─────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    """
    Return accuracy (0-100).

    TODO: Implement this function.
        - Set model to eval mode
        - Use torch.no_grad()
        - Compute and return accuracy as a percentage (0-100)

    Hint: Same logic as evaluate() in train.py.
    """    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total


def get_all_predictions(model, loader, device):
    """Return (y_true, y_pred) lists."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            preds = model(inputs.to(device)).argmax(1)
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(targets.numpy())
    return all_true, all_pred


# ── Training + logging ───────────────────────────────────────────────

def train_cifar100(model, config, run_name):
    """
    Train a model on CIFAR-100 and log to W&B.

    TODO: Complete the parts marked below.  The structure is the same
    as train_and_log() in experiments.py.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Initialise W&B run
    
    wandb.init(project="dl-assignment2", entity="gonzalo-iglesias-deusto-u", name=run_name, config=config)

    
    model = model.to(device)
    wandb.config.update({"num_parameters": count_parameters(model)})
    print(f"[{run_name}] Parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=config.get("batch_size", 128))

    for epoch in range(config["epochs"]):
        # TODO: Implement the training loop for one epoch.
        #   - Set model to train mode
        #   - For each batch: zero_grad → forward → loss → backward → step
        #   - Track total_loss, correct, and total for computing metrics
        #
        # After the inner loop, compute:
        #   train_loss = total_loss / total
        #   train_acc  = 100.0 * correct / total
        #   test_acc   = evaluate(model, test_loader, device)
        #
        # Then log to W&B:
        #   wandb.log({"epoch": epoch, "train/loss": train_loss,
        #              "train/accuracy": train_acc, "test/accuracy": test_acc})
        #
        # Hint: Same training logic as train_one_epoch() in train.py,
        # plus W&B logging.
        model.train()
        total_loss = correct = total = 0

        for images,labels in train_loader:
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()
           
            total_loss += loss.item()
            _, predicted = out.max(1)    
               
            total_loss += loss.item()
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] Loss: {train_loss:.4f} Train: {train_acc:.1f}% Test: {test_acc:.1f}%")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss,
                   "train/accuracy": train_acc, "test/accuracy": test_acc})
        
            
    table = log_prediction_table(model, test_loader, device, num_images=10)
    wandb.log({"predictions": table})
    
    wandb.finish()
    return model


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Part 2 — Transfer Learning on CIFAR-100")
    print("=" * 60)
    
    
    
    # ── Experiment 1: From-scratch CNN ───────────────────────────────
    print("\n── From-scratch CNN on CIFAR-100 ──")
    config_scratch = {
        "architecture": "CNN100-scratch",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 15,
        "dataset": "CIFAR-100",
    }
    model_scratch = CNN100(num_classes=100)
    train_cifar100(model_scratch, config_scratch, run_name="cifar100-scratch")
    # ── Experiment 2: Fine-tuned ResNet-18 ───────────────────────────
    print("\n── Fine-tuned ResNet-18 on CIFAR-100 ──")
    config_resnet = {
        "architecture": "ResNet18-finetune",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 15,
        "dataset": "CIFAR-100",
    }
    model_resnet = create_resnet18(num_classes=100)

    train_cifar100(model_resnet, config_resnet, run_name="cifar100-resnet18")

    
   

    print("\nTransfer learning experiments complete!  Check your W&B dashboard.")
