"""
Part 2: Architecture Comparison on CIFAR-10 with W&B

Assignment 2 - Deep Learning Course, University of Deusto

Run multiple CNN variants and regularisation experiments, logging
everything to Weights & Biases. This script covers Sections 4 and 5
of the assignment (W&B Experiment Tracking + Architecture Comparison).

Usage:
    python experiments.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

# ── Constants ────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

WANDB_PROJECT = "dl-assignment2"       # change if you like


# ── Data ─────────────────────────────────────────────────────────────

def get_data_loaders(batch_size=128, data_dir='./data'):
    """
    Load CIFAR-10 with standard transforms.

    TODO: Implement this function.
        - Training: RandomHorizontalFlip, ToTensor, Normalize(CIFAR10_MEAN, CIFAR10_STD)
        - Test: ToTensor, Normalize(CIFAR10_MEAN, CIFAR10_STD)
        - Use torchvision.datasets.CIFAR10 (download=True)
        - Return (train_loader, test_loader)

    Hint: This is the same loader you wrote in train.py — you can
    import it from there or copy your implementation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader

# ── Configurable CNN ─────────────────────────────────────────────────


    
class ConfigurableCNN(nn.Module):
    """
    A CNN whose depth, width, and regularisation are controlled by a
    config dict.  This lets you run many experiments by just changing
    the config — no need to write a new class each time.

    Supported config keys
    ---------------------
    num_blocks    : int   - number of Conv→ReLU→Pool blocks (1-4)
    filters_start : int   - filters in the first block (doubled each block)
    kernel_size   : int   - convolution kernel size (3 or 5)
    use_batchnorm : bool  - add BatchNorm2d after each Conv
    dropout_rate  : float - dropout before the final linear layer
    num_classes   : int   - output classes (default 10)
    """

    def __init__(self, config):
        super().__init__()

        num_blocks = config.get("num_blocks", 2)
        filters = config.get("filters_start", 16)
        ks = config.get("kernel_size", 3)
        pad = ks // 2                        # same-padding
        use_bn = config.get("use_batchnorm", False)
        num_classes = config.get("num_classes", 10)
        dropout_rate = config.get("dropout_rate", 0.0)


        # TODO: Build the feature-extraction layers dynamically.
        #
        # For each block (0 … num_blocks-1):
        #   - Conv2d(in_channels, out_channels, kernel_size, padding)
        #   - (optionally) BatchNorm2d
        #   - ReLU
        #   - MaxPool2d(2, 2)
        #
        # The first block has in_channels=3.
        # Each subsequent block doubles the filter count.
        #
        # Store the layers in an nn.Sequential called self.features.
        
        layers = []
        in_channels = 3
        out_channels = filters
        
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, ks, padding=pad))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2  

        self.features = nn.Sequential(*layers)

        # Calcular tamaño tras los poolings
        spatial = 32 // (2 ** num_blocks)
        flat_size = in_channels * spatial * spatial


        # TODO: Build the classifier head.
        #
        # After `num_blocks` pooling layers the spatial size is
        # 32 // (2 ** num_blocks) in each dimension.
        # flat_size = final_filters * spatial * spatial
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(flat_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(config.get("dropout_rate", 0.0)),
        #     nn.Linear(128, num_classes),
        # )
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # TODO: pass x through self.features, flatten, then self.classifier
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training helpers ─────────────────────────────────────────────────

def evaluate(model, loader, device):
    
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
    """Return (y_true, y_pred) lists for the full loader."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1)
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(targets.numpy())
    return all_true, all_pred


def log_prediction_table(model, loader, device, num_images=10):
    """Create a wandb.Table with sample images and predictions."""
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    table = wandb.Table(columns=["Image", "True", "Predicted", "Correct"])
    logged = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            preds = outputs.argmax(1).cpu()
            for i in range(inputs.size(0)):
                if logged >= num_images:
                    return table
                img = (inputs[i] * std + mean).clamp(0, 1)
                true_lbl = CIFAR10_CLASSES[targets[i]]
                pred_lbl = CIFAR10_CLASSES[preds[i]]
                table.add_data(wandb.Image(img), true_lbl, pred_lbl,
                               true_lbl == pred_lbl)
                logged += 1
    return table


# ── Main training + logging function ─────────────────────────────────

def train_and_log(config, run_name=None):
    """
    Train a ConfigurableCNN with the given config and log everything
    to W&B: metrics, confusion matrix, and prediction table.

    TODO: Complete the parts marked below.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Initialise W&B ───────────────────────────────────────────
    # TODO: call wandb.init() with project=WANDB_PROJECT, name=run_name,
    #       and config=config.
    wandb.init(project="dl-assignment2", entity="gonzalo-iglesias-deusto-u", name=run_name, config=config)


    # ── 2. Create model ─────────────────────────────────────────────
    model = ConfigurableCNN(config).to(device)
    wandb.config.update({"num_parameters": count_parameters(model)})
    print(f"[{run_name}] Parameters: {count_parameters(model):,}")

    # ── 3. Optimiser & loss ──────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # ── 4. Data ──────────────────────────────────────────────────────
    train_loader, test_loader = get_data_loaders(
        batch_size=config.get("batch_size", 128))

    # ── 5. Training loop ─────────────────────────────────────────────
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = correct = total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        test_acc   = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{config['epochs']}] Loss: {train_loss:.4f} Train: {train_acc:.1f}% Test: {test_acc:.1f}%")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss,
                   "train/accuracy": train_acc, "test/accuracy": test_acc})
        
        

    # ── 6. Log confusion matrix ──────────────────────────────────────
    # TODO: Use get_all_predictions() and wandb.plot.confusion_matrix()
    #       to log a confusion matrix.
    y_true, y_pred = get_all_predictions(model, test_loader, device)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred,
        class_names=list(CIFAR10_CLASSES)
    )})

    # ── 7. Log prediction table ──────────────────────────────────────
    # TODO: Use log_prediction_table() and wandb.log() to log sample
    #       predictions with images.
    table = log_prediction_table(model, test_loader, device, num_images=10)
    wandb.log({"predictions": table})
    
    # ── 8. Finish ────────────────────────────────────────────────────
    wandb.finish()
    return model


# ── Experiment definitions ───────────────────────────────────────────

# Each dict below defines one W&B run.  You must run the baseline plus
# at least 3 architectural variants and 2 regularisation experiments.
#
# Feel free to add more, change hyperparameters, or combine techniques.

EXPERIMENTS = {
    # ── Baseline (same SimpleCNN from Part 1) ──
    "baseline": {
        "num_blocks": 2,
        "filters_start": 16,
        "kernel_size": 3,
        "use_batchnorm": False,
        "dropout_rate": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 0,
        "batch_size": 128,
        "epochs": 10,
    },

    # ── Architectural variants (pick ≥ 3) ──

     "deeper": {
        "num_blocks": 3, "filters_start": 16, "kernel_size": 3,
        "use_batchnorm": False, "dropout_rate": 0.0,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "grad_clip": 0, "batch_size": 128, "epochs": 10,
    },
    "shallower": {
        "num_blocks": 1, "filters_start": 16, "kernel_size": 3,
        "use_batchnorm": False, "dropout_rate": 0.0,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "grad_clip": 0, "batch_size": 128, "epochs": 10,
    }, "large-kernel": {
        "num_blocks": 2, "filters_start": 16, "kernel_size": 5,
        "use_batchnorm": False, "dropout_rate": 0.0,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "grad_clip": 0, "batch_size": 128, "epochs": 10,
    },
    # ── Regularization variants (pick ≥ 2) ──

    "dropout": {
        "num_blocks": 2, "filters_start": 16, "kernel_size": 3,
        "use_batchnorm": False, "dropout_rate": 0.5,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "grad_clip": 0, "batch_size": 128, "epochs": 10,
    },
    "l2": {
        "num_blocks": 2, "filters_start": 16, "kernel_size": 3,
        "use_batchnorm": False, "dropout_rate": 0.0,
        "learning_rate": 0.001, "weight_decay": 1e-3,
        "grad_clip": 0, "batch_size": 128, "epochs": 10,
    },
}


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Part 2 — Architecture Comparison on CIFAR-10")
    print("=" * 60)

    for name, cfg in EXPERIMENTS.items():
        print(f"\n{'─' * 60}")
        print(f"Running experiment: {name}")
        print(f"{'─' * 60}")
        train_and_log(cfg, run_name=name)

    print("\nAll experiments complete!  Check your W&B dashboard.")
