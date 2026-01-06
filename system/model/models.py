# -*- coding: utf-8 -*-
# @Time : 2025/11/27
# @Author : Yang


from torch import nn
import torch.nn.functional as F
from torchinfo import summary


# ====================================================================================================================
class CNNMNIST_LIT(nn.Module):
    def __init__(self,num_classes=10):
        super(CNNMNIST_LIT, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x



# ====================================================================================================================
class CNNMNIST_BIG(nn.Module):
    def __init__(self,num_classes=10):
        super(CNNMNIST_BIG, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc(x)
        return x


# ====================================================================================================================
class CNNEMNIST(nn.Module):
    def __init__(self, num_classes=62):
        super(CNNEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================
class CNNCIFAR(nn.Module):
    def __init__(self,num_classes=10):
        super(CNNCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc = nn.Linear(500, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x




if __name__ == '__main__':
    print('\n', '*' * 50, 'LeNet5', '*' * 50, '\n')
    mnist = LeNet5()
    for name, parameter in mnist.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist, input_size=(1, 1, 28, 28))

    print('\n', '*' * 50, 'MNIST LIT', '*' * 50, '\n')
    mnist = CNNMNIST_LIT()
    for name, parameter in mnist.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist, input_size=(1, 1, 28, 28))

    print('\n', '*' * 50, 'MNIST BIG', '*' * 50, '\n')
    mnist = CNNMNIST_BIG()
    for name, parameter in mnist.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist, input_size=(1, 1, 28, 28))

    print('\n', '*' * 50, 'EMNIST', '*' * 50, '\n')
    emnist = CNNEMNIST()
    for name, parameter in emnist.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(emnist, input_size=(1, 1, 28, 28))

    print('\n', '*' * 50, 'CIFAR10', '*' * 50, '\n')
    cifar3 = CNNCIFAR()
    for name, parameter in cifar3.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(cifar3, input_size=(1, 3, 32, 32))
