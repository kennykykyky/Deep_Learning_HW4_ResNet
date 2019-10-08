import torch
import torch.nn as nn
from torchvision import models
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import time
import copy
from random import randint
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from net import ResNet
from Cal_Accuracy import Cal_Accuarcy

parser = argparse.ArgumentParser()
parser.add_argument('--is_gpu', type = bool, default = True)
parser.add_argument('--batch', type = bool, default = 64)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--wd', type = float, default = 5e-4)
parser.add_argument('--epochs', type=int, default= 50,
                    help='number of epochs to train')
opt = parser.parse_args()

# load dataset CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root = './',
                                          train = True,
                                          transform = transforms.ToTensor(),
                                          download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './',
                                          train = False,
                                          transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = opt.batch,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = opt.batch,
                                          shuffle = False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet()

# define optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wd)

if opt.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

start_epoch = 0

# train dataset
for epoch in range(start_epoch, opt.epochs + start_epoch):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if opt.is_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)

    # Calculate training/test set accuracy of the existing model
    train_accuracy = calculate_accuracy(train_loader, opt.is_gpu)
    test_accuracy = calculate_accuracy(test_loader, opt.is_gpu)

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))


print('==> Finished Training ...')
