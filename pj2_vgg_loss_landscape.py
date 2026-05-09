import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
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
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(3))

# Initialize your data loader and make sure that dataloader works as expected by observing one sample from it.
train_loader = get_cifar_loader(train=True, n_items=1000)  # 使用部分数据集加快训练速度
val_loader = get_cifar_loader(train=False, n_items=1000)

for X, y in train_loader:
    print(X[0])
    print(y[0])
    print(X[0].shape)
    img = np.transpose(X[0], [1, 2, 0])
    plt.imshow(img * 0.5 + 0.5)
    plt.savefig('sample.png')
    print(X[0].max())
    print(X[0].min())
    break

# Calculate the accuracy of model classification
def get_accuracy():
    pass

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
        grad = []  # Record the loss gradient of each step
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
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
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

# Train your model and record loss landscape
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), losses, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve
min_curve = []
max_curve = []
for loss_list in losses:
    min_curve.append(min(loss_list))
    max_curve.append(max(loss_list))

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
    plt.show()

plot_loss_landscape(max_curve, min_curve, 'VGG_A Loss Landscape')

# Repeat the process for VGG_A_BatchNorm
model_bn = VGG_A_BatchNorm()
optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
losses_bn, grads_bn = train(model_bn, optimizer_bn, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss_bn.txt'), losses_bn, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads_bn.txt'), grads_bn, fmt='%s', delimiter=' ')

min_curve_bn = []
max_curve_bn = []
for loss_list in losses_bn:
    min_curve_bn.append(min(loss_list))
    max_curve_bn.append(max(loss_list))

plot_loss_landscape(max_curve_bn, min_curve_bn, 'VGG_A_BatchNorm Loss Landscape')

# Combined plot
plt.figure()
plt.fill_between(range(len(max_curve)), min_curve, max_curve, alpha=0.3, label='Base Model')
plt.fill_between(range(len(max_curve_bn)), min_curve_bn, max_curve_bn, alpha=0.3, label='BatchNorm Model')
plt.plot(max_curve, label='Base Model Max Curve')
plt.plot(min_curve, label='Base Model Min Curve')
plt.plot(max_curve_bn, label='BatchNorm Model Max Curve')
plt.plot(min_curve_bn, label='BatchNorm Model Min Curve')
plt.title('Loss Landscape Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show() 
