import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
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
from utils import progress_bar, plot_accuracies, plot_losses, plot_lrs

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
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
X_train, X_valid, y_train, y_valid = train_test_split(train_images_tensor, train_labels_tensor, test_size=0.2, random_state=42)
print("all test images", len(all_test_images))
print("all test labels", len(all_test_labels))
print("test image tensor", len(test_images_tensor))
print("test images tensor", len(test_labels_tensor))
# Training dataset
train_dataset = CustomTensorDataset(tensors=(X_train, y_train), transform=get_transform("train"))
valid_dataset = CustomTensorDataset(tensors=(X_valid, y_valid), transform=get_transform("valid"))
batch_size =  128
train_dataset = CustomTensorDataset(tensors=(train_images_tensor, train_labels_tensor), transform=get_transform("train"))
# batch_size =  32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print("train loader length: ", len(trainloader))
# Testing dataset
test_dataset = CustomTensorDataset(tensors=(test_images_tensor, test_labels_tensor), transform = get_transform("test"))
batch_size =  100
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("test loader length: ", len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# print('==> Building model..')      
net = ResNet5M()
# net = ResNet5MWithAttention()
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


initial_lr = 0.01
final_lr = 0.001
total_epochs = 200
def lr_lambda(epoch):
    return 1 - (epoch / total_epochs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 
                        # lr=args.lr,
                        lr=initial_lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = LambdaLR(optimizer, lr_lambda)

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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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

for epoch in range(start_epoch, start_epoch+200):
    # print(epoch)
    train(epoch)
    valid(epoch)
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
    if epoch == 159:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions159.csv")
        print("over")
    if epoch == 169:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions169.csv")
        print("over")
    if epoch == 179:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions179.csv")
        print("over")
    if epoch == 189:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions189.csv")
        print("over")
    if epoch == 199:
        predictions = generate_predictions(net, testloader)
        save_predictions_to_csv(predictions, list(range(len(predictions))), csv_filename="predictions199.csv")
        print("over")


plot_accuracies()

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

    