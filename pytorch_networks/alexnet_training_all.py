import torch
import torchvision
import numpy as np
import pickle
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="7"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4096, 384)
        self.fc2 = nn.Linear(384,192)
        self.fc3 = nn.Linear(192,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.CrossEntropyLoss()

batch_size = [256, 2000]
lrs = [1e-4, 1e-3]
momentums = [0.9, 0.99]
n_epochs = [150,500]
interval = [50,5]
reps = 5

for i in range(2):
    #trainset = list(load_data(train_files))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size[i], shuffle=True)

    #testset = list(load_data(test_files))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size[i], shuffle=True)


    for r in range(reps):
        net = Net()
        net = net.cuda()
        criterion = criterion.cuda()

        optimizer = optim.SGD(net.parameters(), lr=lrs[i], momentum=momentums[i])
        if i==0:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)

        for epoch in range(n_epochs[i]):
            running_loss = 0.0
            for j, data in enumerate(trainloader,0):
                inputs, labels = data[0].cuda(), data[1].cuda()
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data
                if (j+1)%interval[i]==0:
                    print('[%d, %5d] loss: %.5f' % (epoch+1,j+1, running_loss/interval[i]))
                    running_loss = 0.0
            if i==0:
                scheduler.step()

        print('Finished Training Rep %d with batch size %d' % (r+1, batch_size[i]))

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data,1)
        print('Predicted: ' + ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

        correct = 0
        total = 0
        for data in trainloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        print('Accuracy of the network on the 50000 train images: %f %%' % (100. * correct / total))

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print('Accuracy of the network on the 10000 test images: %f %%' % (100. * correct / total))

        layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        weights, biases = [],[]
        for l in layers:
            weights.append(net.__getattr__(l).weight.data.cpu().numpy())
            biases.append(net.__getattr__(l).bias.data.cpu().numpy())

        net_weights = {'weights':weights, 'biases':biases}
        with open('alexnet_weights_%d_rep_%d.pkl'%(batch_size[i], r+1),'wb') as f:
            pickle.dump(net_weights, f)
