import torch
import torch.optim as optim
from tqdm import tqdm
from pj2_data_loaders import get_cifar_loader
from pj2_vgg import VGG_A, VGG_A_BatchNorm

def train(model, train_loader, criterion, optimizer, device, epoch, losses):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = outputs
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(test_loader), accuracy

if __name__ == '__main__':
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    # 加载部分数据集
    train_loader = get_cifar_loader(train=True, n_items=500)
    test_loader = get_cifar_loader(train=False, n_items=500)

    # 模型、激活函数、损失函数和优化器的设置
    models = {'BN Model': VGG_A_BatchNorm}
    activations = ['relu', 'leaky_relu', 'sigmoid']
    loss_functions = [torch.nn.CrossEntropyLoss()]
    optimizers = {
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adam': optim.Adam
    }
    regularization = 0.01  # L2正则化
    num_epochs = 5  # 减少训练轮数

    for model_name, model_class in models.items():
        for activation in activations:
            for criterion in loss_functions:
                for optim_name, optim_func in optimizers.items():
                    model = model_class(activation=activation).to(device)
                    optimizer = optim_func(model.parameters(), lr=0.001, weight_decay=regularization)
                    losses = []
                    for epoch in range(num_epochs):
                        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, losses)
                        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
                        print(f'{model_name} - Activation: {activation}, Optimizer: {optim_name}, Loss Function: {criterion}, Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
                    torch.save(model.state_dict(), f'{model_name}_{activation}_{optim_name}_{criterion}.pth') 
