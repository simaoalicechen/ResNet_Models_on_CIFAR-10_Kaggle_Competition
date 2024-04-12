import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from models.resnet import ResNet18, ResNet5M, ResNet5MWithDropout, ResNet2_Modified, ResNet5M2Layers, ResNet34
import matplotlib.pyplot as plt
from customTensorDataset import CustomTensorDataset, get_transform, test_unpickle
import os
import argparse
import pickle
from models import *
from utils import progress_bar, plot_losses, plot_acc, get_lrs, plot_lr

# Parser 
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0 

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# Data
print('==> Preparing data..')

# Getting training and validating data: 
cifar10_dir = 'data/cifar-10-batches-py'
meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
label_names = meta_data_dict[b'label_names']
all_images = []
all_labels = []
for i in range(1, 6):
    batch_dict = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
    batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
    batch_labels = batch_dict[b'labels']
    all_images.append(batch_images)
    all_labels.append(batch_labels)
    train_images_tensor = torch.Tensor(np.concatenate(all_images, axis=0)).to(device)
    train_labels_tensor = torch.Tensor(np.concatenate(all_labels, axis=0)).to(torch.long).to(device)
print("all_images", len(all_images))
print("all_labels", len(all_labels))
print("train_images_tensor", len(train_images_tensor ))
print("train_labels_tensor", len(train_labels_tensor))
# Getting test data here: 
all_test_images = []
all_test_labels = []
batch_dict = test_unpickle('cifar_test_nolabels.pkl')
batch_test_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
batch_test_labels = batch_dict[b'ids']
all_test_images.append(batch_test_images)
all_test_labels.append(batch_test_labels)
test_images_tensor = torch.Tensor(np.concatenate(all_test_images, axis=0)).to(device)
test_labels_tensor = torch.Tensor(np.concatenate(all_test_labels, axis=0)).to(torch.long).to(device)
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
X_train, X_valid, y_train, y_valid = train_test_split(train_images_tensor, train_labels_tensor, test_size=0.1, random_state=42)
print("all test images", len(all_test_images))
print("all test labels", len(all_test_labels))
print("test image tensor", len(test_images_tensor))
print("test images tensor", len(test_labels_tensor))
# Training dataset
train_dataset = CustomTensorDataset(tensors=(X_train, y_train), transform=get_transform("train"))
valid_dataset = CustomTensorDataset(tensors=(X_valid, y_valid), transform=get_transform("valid"))
batch_size =  128
train_dataset = CustomTensorDataset(tensors=(train_images_tensor, train_labels_tensor), transform=get_transform("train"))
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print("train loader length: ", len(trainloader))
# Testing dataset
test_dataset = CustomTensorDataset(tensors=(test_images_tensor, test_labels_tensor), transform = get_transform("test"))
test_batch_size =  100
testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
print("test loader length: ", len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
  
net = ResNet5M()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_dir = './checkpoint/'
os.makedirs(checkpoint_dir, exist_ok=True)

summary(net, input_size = (400, 3, 32, 32))
print("Trainable Parameters: "+ str(summary(net, input_size = (400, 3, 32, 32)).trainable_params))

checkpoint_path = './checkpoint/ckpt_epoch.pth'

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])

        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] + 1
        print('==> Resuming from checkpoint..')
    except FileNotFoundError:
        print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
else:
    print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
print(start_epoch)
print("best_acc", best_acc)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_value_(list(net.parameters()), grad_clip)

        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_trend.append(current_lr)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'train Loss: %.3f | train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_accuracy = 100.0* correct/total
    train_loss /= len(trainloader)

    train_loss_trend.append(train_loss)
    train_acc_trend.append(train_accuracy)
    # Save training checkpoint after each epoch
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
    }, './checkpoint/ckpt_epoch{}.pth'.format(epoch))


def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    valid_accuracy = 100.0 * correct / total
    test_loss /= len(validloader)
    valid_loss_trend.append(test_loss)
    valid_acc_trend.append(valid_accuracy)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc

    checkpoint = {
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_loss_trend': train_loss_trend,
        'valid_loss_trend': valid_loss_trend,
        'train_acc_trend': train_acc_trend,
        'valid_acc_trend': valid_acc_trend,
    }

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
    torch.save(checkpoint, f'./checkpoint/ckpt{epoch}.pth')

            
def generate_predictions(model, test_loader):
    model.eval()  
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images, _ = batch 
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())  
    return predictions

def save_predictions_to_csv(predictions, test_ids, csv_filename="predictions.csv"):
    df = pd.DataFrame({"ID": test_ids, "Labels": predictions})
    df.to_csv(csv_filename, index=False)
    print(f"Predictions saved to {csv_filename}")


epochs = 300
max_lr = 0.1
#grad_clip = 0.1
weight_decay_adam = 5e-4
opt_func = torch.optim.SGD
criterion = nn.CrossEntropyLoss()
optimizer = opt_func(net.parameters(), max_lr, weight_decay=weight_decay_adam)
scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# """
# TODO

# ALWAYS record your changes, we don't have to rerun them. Save everything. 
# Any combinations are fine, as long as you can have a reasoning behind it. 
#     """

batch_size_para = batch_size 
lr_para = "OneCycleLR"
scheduler_para = "WD: " + str(weight_decay_adam)
lr_para = "Max LR: " + str(max_lr)
grad_clip_para = "GC: " + str(grad_clip)
opt_para = "SGD"
epoch_para = "Epochs: " + str(epochs)
paras_for_graph = [lr_para, scheduler_para, grad_clip_para, opt_para, epoch_para, lr_para]

train_loss_trend = []
train_acc_trend = []
valid_loss_trend = []
valid_acc_trend = []
lr_trend = []

for epoch in range(epochs):
    train(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    lr_trend.append(current_lr)
    valid(epoch)
    scheduler.step()

predictions = generate_predictions(net, testloader)
save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions_final.csv")
print("checking progress")
print(train_acc_trend)
print(train_loss_trend)
print(valid_acc_trend)
print(valid_loss_trend)
print("over")
plot_losses(train_loss_trend, valid_loss_trend, epoch = epoch, hyperparam = paras_for_graph)
plot_acc(train_acc_trend, valid_acc_trend, epoch = epoch, hyperparam = paras_for_graph)
plot_lr(lr_trend, epoch = epoch, hyperparam = paras_for_graph)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def load_dataset(split, augment=False):
    if split == "real_cifar":
        cifar10_dir = 'data/cifar-10-batches-py'
        meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
        label_names = meta_data_dict[b'label_names']
        all_images = []
        all_labels = []
        batch_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
        batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
        batch_labels = batch_dict[b'labels']
        all_images.append(batch_images)
        all_labels.append(batch_labels)
        real_test_images_tensor = torch.Tensor(np.concatenate(all_images, axis=0)).to(device)
        real_test_labels_tensor = torch.Tensor(np.concatenate(all_labels, axis=0)).to(torch.long).to(device)
        return real_test_images_tensor, real_test_labels_tensor

def get_real_test_dataloader():
    print("printing actual cifar10 test dataloader")
    device = get_default_device()
    test_data, test_labels = load_dataset("real_cifar", augment=False)
    test_dataset = CustomTensorDataset(tensors=(test_data, test_labels), transform = get_transform("test"))
    batch_size = 400
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def test_acc(model, testLoader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testLoader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))

    valid_accuracy = 100.0 * correct / total

    # Save checkpoint.
    acc = 100.*correct/total
    return acc



#Cifar10 Test Set Results
realTestDataLoader = get_real_test_dataloader()
realTestDataLoader = DeviceDataLoader(realTestDataLoader, device)
real_results = test_acc(net, realTestDataLoader)
print(real_results)





    