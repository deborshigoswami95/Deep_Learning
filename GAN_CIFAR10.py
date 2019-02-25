import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as fnn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import time

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3, 196, 3, stride=1, padding=1)
        self.layer_norm_1 = nn.LayerNorm([196,32,32])
        self.conv2 = nn.Conv2d(196, 196, 3, stride=2, padding=1)
        self.layer_norm_2 = nn.LayerNorm([196,16,16])
        self.conv3 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.layer_norm_3 = nn.LayerNorm([196,16,16])
        self.conv4 = nn.Conv2d(196, 196, 3, stride=2, padding=1)
        self.layer_norm_4 = nn.LayerNorm([196,8,8])
        self.conv5 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.layer_norm_5 = nn.LayerNorm([196,8,8])
        self.conv6 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.layer_norm_6 = nn.LayerNorm([196,8,8])
        self.conv7 = nn.Conv2d(196, 196, 3, stride=1, padding=1)
        self.layer_norm_7 = nn.LayerNorm([196,8,8])
        self.conv8 = nn.Conv2d(196, 196, 3, stride=2, padding=1)
        self.layer_norm_8 = nn.LayerNorm([196,4,4])
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding = 0)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196,10)
        
    def forward(self,x, extract_features = 0):
        x = fnn.leaky_relu(self.layer_norm_1(self.conv1(x)))
        x = fnn.leaky_relu(self.layer_norm_2(self.conv2(x)))
        x = fnn.leaky_relu(self.layer_norm_3(self.conv3(x)))
        x = fnn.leaky_relu(self.layer_norm_4(self.conv4(x)))
        x = fnn.leaky_relu(self.layer_norm_5(self.conv5(x)))
        x = fnn.leaky_relu(self.layer_norm_6(self.conv6(x)))
        x = fnn.leaky_relu(self.layer_norm_7(self.conv7(x)))
        x = fnn.leaky_relu(self.layer_norm_8(self.conv8(x)))
        if(extract_features==8):
            x = fnn.max_pool2d(x,4,4)
            x = x.view(-1, 196)
            return x 
        x = self.maxpool(x)
        x = x.reshape(x.size(0),-1)
        disc_out = self.fc1(x)
        class_out = self.fc10(x)
        return disc_out, class_out
        
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(100,196*4*4)
        self.conv1 = nn.ConvTranspose2d(196,196,4,stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(196,196,3,stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(196,196,3,stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(196,196,3,stride = 1, padding = 1)
        self.conv5 = nn.ConvTranspose2d(196,196,4,stride = 2, padding = 1)
        self.conv6 = nn.Conv2d(196,196,3,stride = 1, padding = 1)
        self.conv7 = nn.ConvTranspose2d(196,196,4,stride = 2, padding = 1)
        self.conv8 = nn.Conv2d(196,3,3,stride = 1,padding = 1)
        self.bn1 = nn.BatchNorm1d(196 * 4 * 4)
        self.bn = nn.BatchNorm2d(196)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        x = self.bn1(self.fc1(x))
        x = x.view(-1, 196, 4, 4)
        x = fnn.relu(self.bn(self.conv1(x)))
        x = fnn.relu(self.bn(self.conv2(x)))
        x = fnn.relu(self.bn(self.conv3(x)))
        x = fnn.relu(self.bn(self.conv4(x)))
        x = fnn.relu(self.bn(self.conv5(x)))
        x = fnn.relu(self.bn(self.conv6(x)))
        x = fnn.relu(self.bn(self.conv7(x)))
        x = self.tanh(self.conv8(x))
        return x
        
        


batch_size=128

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

aD =  Discriminator()
aD.cuda()

aG = Generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

n_z = 100
n_classes = 10
gen_train = 1

np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

start_time = time.time()





num_epochs = 200

# Train the model
for epoch in range(0,num_epochs):
    aG.train()
    aD.train()
    
    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []
    
    
    
                    
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        
        if(Y_train_batch.shape[0] < batch_size):
            continue
        # train G
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()
            
            for group in optimizer_g.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000

            optimizer_g.step()
    
        # train D
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()
        
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        optimizer_d.step()

        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                    "%.2f" % np.mean(loss2), 
                                    "%.2f" % np.mean(loss3), 
                                    "%.2f" % np.mean(loss4), 
                                    "%.2f" % np.mean(loss5), 
                                    "%.2f" % np.mean(acc1))

    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test, time.time()-start_time)

    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if(((epoch+1)%1)==0):
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')
    
    
torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')


        
        



model =  Discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def step_size_calc(epoch):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

epochs = 100
for epoch in range(epochs):
    step_size_calc(epoch)
    
    
                    
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        
        loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        
    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test, time.time()-start_time)
        

torch.save(model,'cifar10.model')