import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as fnn
import torch.optim as optim

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 64, 4, stride=1, padding=2)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.FC1 = nn.Linear(64*4*4, 500)
        self.FC2 = nn.Linear(500, 500)
        self.FC3 = nn.Linear(500, 10)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d()
        
    def forward(self,x):
        x = self.bn(self.conv1(x))
        x = self.dropout(self.maxpool(self.conv2(x)))
        x = self.bn(self.conv3(x))
        x = self.dropout(self.maxpool(self.conv4(x)))
        x = self.bn(self.conv5(x))
        x = self.dropout(self.conv6(x))
        x = self.bn(self.conv7(x))
        x = self.bn(self.dropout(self.conv8(x)))
        x = x.view(-1, 64*4*4)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                 transforms.RandomVerticalFlip(), 
                                 transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root='D:\Downloads',train=True, transform=transforms,download=True)
test_data = torchvision.datasets.CIFAR10(root='D:\Downloads',train=False, transform=transforms)
 

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size= 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size= 64, shuffle=True)



model = net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 150


for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
    print("We are at epoch {}".format(epoch))
    
    if epoch % 5 == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test Accuracy: %d %%' % (100 * correct / total))

print('\n\n\nFinished Training\n\n\n')

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy: %d %%' % (
    100 * correct / total))

