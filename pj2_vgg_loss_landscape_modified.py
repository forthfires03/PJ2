import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from IPython import display

from pj2_vgg import VGG_A, VGG_A_BatchNorm
from pj2_data_loaders import get_cifar_loader

# Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128

# Add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Train function with loss landscape recording
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # Record the loss value of each step
        grad_list = []  # Record the loss gradient of each step
        learning_curve[epoch] = 0  # Maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            learning_curve[epoch] += loss.item()
            loss_list.append(loss.item())
            loss.backward()

            # Record gradients
            for param in model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.clone().detach().cpu().numpy().flatten())

            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad_list)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for val_data in val_loader:
                x_val, y_val = val_data
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_prediction = model(x_val)
                val_loss += criterion(val_prediction, y_val).item()
                _, predicted = torch.max(val_prediction, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
            val_accuracy = 100 * correct / total
            val_accuracy_curve[epoch] = val_accuracy

        print(f'Epoch {epoch+1}/{epochs_n}, Train Loss: {learning_curve[epoch]}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {val_accuracy}')

    return losses_list, grads

# Training for different learning rates
def train_with_different_lrs(model_class, model_name):
    all_losses = []
    all_grads = []
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    for lr in learning_rates:
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
        all_losses.append(losses)
        all_grads.append(grads)
        np.savetxt(os.path.join(loss_save_path, f'loss_{model_name}_lr{lr}.txt'), np.array(losses), fmt='%s', delimiter=' ')
        # Convert gradients to a 2D array for saving
        grad_array = np.array([np.hstack(g) for g in grads])
        np.savetxt(os.path.join(grad_save_path, f'grads_{model_name}_lr{lr}.txt'), grad_array, fmt='%s', delimiter=' ')
    return all_losses, all_grads

# Maintain two lists: max_curve and min_curve
def compute_max_min_curve(losses):
    max_curve = []
    min_curve = []
    for loss_list in zip(*losses):
        max_curve.append(max([max(l) for l in loss_list]))
        min_curve.append(min([min(l) for l in loss_list]))
    return max_curve, min_curve

# Plot the final loss landscape
def plot_loss_landscape(max_curve, min_curve, title):
    plt.figure()
    plt.fill_between(range(len(max_curve)), min_curve, max_curve, alpha=0.3)
    plt.plot(max_curve, label='Max Curve')
    plt.plot(min_curve, label='Min Curve')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(fig_save_path, f"{title.replace(' ', '_')}.png"))
    plt.close()

if __name__ == '__main__':
    set_random_seeds(seed_value=2020, device=device)
    loss_save_path = './losses'
    grad_save_path = './grads'
    fig_save_path = './figures'
    os.makedirs(loss_save_path, exist_ok=True)
    os.makedirs(grad_save_path, exist_ok=True)
    os.makedirs(fig_save_path, exist_ok=True)

    # 加载部分数据集
    train_loader = get_cifar_loader(train=True, n_items=500)
    val_loader = get_cifar_loader(train=False, n_items=500)

    # Train models
    losses_base, grads_base = train_with_different_lrs(VGG_A, 'VGG_A')
    losses_bn, grads_bn = train_with_different_lrs(VGG_A_BatchNorm, 'VGG_A_BatchNorm')

    max_curve_base, min_curve_base = compute_max_min_curve(losses_base)
    max_curve_bn, min_curve_bn = compute_max_min_curve(losses_bn)

    plot_loss_landscape(max_curve_base, min_curve_base, 'VGG_A Loss Landscape')
    plot_loss_landscape(max_curve_bn, min_curve_bn, 'VGG_A_BatchNorm Loss Landscape')

    # Combined plot
    plt.figure()
    plt.fill_between(range(len(max_curve_base)), min_curve_base, max_curve_base, alpha=0.3, label='Base Model')
    plt.fill_between(range(len(max_curve_bn)), min_curve_bn, max_curve_bn, alpha=0.3, label='BatchNorm Model')
    plt.plot(max_curve_base, label='Base Model Max Curve')
    plt.plot(min_curve_base, label='Base Model Min Curve')
    plt.plot(max_curve_bn, label='BatchNorm Model Max Curve')
    plt.plot(min_curve_bn, label='BatchNorm Model Min Curve')
    plt.title('Loss Landscape Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(fig_save_path, "Loss_Landscape_Comparison.png"))
    plt.close() 
