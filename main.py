import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from models.resnet import ResNet18, ResNet5M
from customTensorDataset import CustomTensorDataset, get_transform, test_unpickle
import os
import argparse
import pickle
from models import *
from utils import progress_bar

import torch

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

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

# Getting test data here: 
all_test_images = []
all_test_labels = []
batch_dict = test_unpickle('cifar_test_nolabels.pkl')
batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
batch_labels = batch_dict[b'ids']
all_test_images.append(batch_images)
all_test_labels.append(batch_labels)
test_images_tensor = torch.Tensor(np.concatenate(all_test_images, axis=0)).to(device)
test_labels_tensor = torch.Tensor(np.concatenate(all_test_labels, axis=0)).to(torch.long).to(device)
print(len(all_test_images))
print(len(all_test_labels))
print(len(test_images_tensor))
print(len(test_labels_tensor))
# Training dataset
train_dataset = CustomTensorDataset(tensors=(train_images_tensor, train_labels_tensor), transform=get_transform("train"))
batch_size = 128
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Testing dataset
test_data, test_labels = load_dataset("test", augment = False)
test_dataset = CustomTensorDataset(tensors=(test_images_tensor, test_labels_tensor), transform = get_transform("test"))
batch_size = 400
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')      
net = ResNet5M()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# import os

# checkpoint_dir = './checkpoint/'
# os.makedirs(checkpoint_dir, exist_ok=True)

# summary(net, input_size = (400, 3, 32, 32))
# print("Trainable Parameters: "+ str(summary(net, input_size = (400, 3, 32, 32)).trainable_params))

# checkpoint_path = './checkpoint/ckpt.pth'

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Resuming from checkpoint..')
    except FileNotFoundError:
        print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
else:
    print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'train Loss: %.3f | train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'test Loss: %.3f | test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

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

for epoch in range(start_epoch, start_epoch+210):
    print(epoch)
    train(epoch)
    test(epoch)
    scheduler.step()
    if epoch == 9:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions9.csv")
        print("over")
    if epoch == 59:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions59.csv")
        print("over")
    if epoch == 109:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions109.csv")
        print("over")
    if epoch == 189:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions189.csv")
        print("over")
    if epoch == 209:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions209.csv")
        print("over")

# if __name__ == "__main__":
#     net = ResNet5M()
#     net = net.to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True
#     net.load_state_dict(torch.load("checkpoint.pth"))
#     test(epoch)
#     # scheduler.step()
#         # if epoch == 29:
#     predictions = generate_predictions(net, testloader)
#     save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions.csv")
#     print("over")

# if __name__ == "__main__":
#     net = ResNet5M()
#     net = net.to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True
#     net.load_state_dict(torch.load("checkpoint.pth"))
#     for epoch in range(start_epoch, start_epoch+30):
#         train(epoch)
#         test(epoch)
#         scheduler.step()
#         if epoch == 29:
#             predictions = generate_predictions(net, testloader)
#             save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions.csv")
#             print("over")

    