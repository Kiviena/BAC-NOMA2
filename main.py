import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

CHANNEL_CLIP = 1
DISTANCE = 50
ANTENNA = 5
ALPHA = 2
Y = 1
SIGMA = 5

MAX_EPISODES = 7000
BATCH_SIZE = 50
DATA_SIZE = 50000
LR = 0.01

class Channel:
    @classmethod
    def rayleigh(cls, dim, distance, alpha):
        channel = (np.sqrt(0.5)*np.clip(np.random.randn(dim), -CHANNEL_CLIP, CHANNEL_CLIP)+
                   np.sqrt(0.5)*1j*np.clip(np.random.randn(dim), -CHANNEL_CLIP, CHANNEL_CLIP))/np.sqrt(distance**alpha)
        return channel

    @classmethod
    def rayleigh_2d(cls, dim1, dim2, distance, alpha):
        channel = (np.sqrt(0.5)*np.clip(np.random.standard_normal((dim1, dim2)), -CHANNEL_CLIP, CHANNEL_CLIP) +
                           np.sqrt(0.5)*1j*np.clip(np.random.standard_normal((dim1, dim2)), -CHANNEL_CLIP, CHANNEL_CLIP)/np.sqrt(distance**alpha))
        return channel

    @classmethod
    def ricean(cls, dim, K, distance, alpha):
        rayleigh = np.sqrt(0.5)*np.clip(np.random.randn(dim), -CHANNEL_CLIP, CHANNEL_CLIP)+ np.sqrt(0.5)*1j*np.clip(np.random.randn(dim), -CHANNEL_CLIP, CHANNEL_CLIP)
        channel = (np.sqrt(K/(K+1)) + np.sqrt(1/(K+1))*rayleigh)/np.sqrt(distance**alpha)
        return channel

    @classmethod
    def ricean_2d(cls, dim1, dim2, K, distance, alpha):
        rayleigh_2d = np.sqrt(0.5)*np.clip(np.random.standard_normal((dim1, dim2)), -CHANNEL_CLIP, CHANNEL_CLIP) + np.sqrt(0.5)*1j*np.clip(np.random.standard_normal((dim1, dim2)), -CHANNEL_CLIP, CHANNEL_CLIP)
        channel = (np.sqrt(K/(K+1)) + np.sqrt(1/(K+1))*rayleigh_2d)/np.sqrt(distance**alpha)
        return channel

class Network(nn.Module):
    def __init__(self, input, output):
        super().__init__()

        self.input = nn.Linear(input, 256)
        self.hidden1 = nn.Linear(256,512)
        self.hidden2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class MyLoss (nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, input, channel):
        w1_real = input[:,:ANTENNA]
        w1_img = input[:,ANTENNA:2*ANTENNA]
        w2_real = input[:,2*ANTENNA:3*ANTENNA]
        w2_img = input[:,3*ANTENNA:]

        sinr1 = (torch.sum((w1_real*channel[:,:ANTENNA] - w1_img*channel[:, ANTENNA:2*ANTENNA]), 1)**2 + torch.sum((w1_real*channel[:, ANTENNA:2*ANTENNA] + w1_img*channel[:,:ANTENNA]), 1)**2)/(Y*SIGMA+SIGMA)
        temp1 = torch.sum((w2_real*channel[:,2*ANTENNA:3*ANTENNA] - w2_img*channel[:,3*ANTENNA:4*ANTENNA]), 1)**2 + torch.sum((w2_real*channel[:,3*ANTENNA:4*ANTENNA] + w2_img*channel[:,2*ANTENNA:3*ANTENNA]), 1)**2  # |h2w2|^2
        temp2 = torch.sum((w1_real*channel[:,2*ANTENNA:3*ANTENNA] - w1_img*channel[:,3*ANTENNA:4*ANTENNA]), 1)**2 + torch.sum((w1_real*channel[:,3*ANTENNA:4*ANTENNA] + w1_img*channel[:,2*ANTENNA:3*ANTENNA]), 1)**2  # |h2w1|^2
        sinr2 = temp1/(temp2 + channel[:,-1]*Y*SIGMA + SIGMA)
        loss = -torch.mean(torch.log2(1+sinr1)+torch.log2(1+sinr2))
        return loss

#---------------------generate dataset----------------------------
channel = []
for i in range(DATA_SIZE):
    h1 = Channel.ricean(dim=ANTENNA, K=1, distance=DISTANCE, alpha=ALPHA)
    h2 = Channel.ricean(dim=ANTENNA, K=1, distance=DISTANCE, alpha=ALPHA)
    # hbd = Channel.ricean(dim=ANTENNA, K=1, distance=DISTANCE, alpha=ALPHA)
    g1 = Channel.ricean(dim=1, K=1, distance=DISTANCE, alpha=ALPHA)
    g2 = Channel.ricean(dim=1, K=1, distance=DISTANCE, alpha=ALPHA)
    channel.append(np.hstack((np.real(h1), -1*np.imag(h1), np.real(h2), -1*np.imag(h2), (np.linalg.norm(g2)/np.linalg.norm(g1))**2)))
channel = np.array(channel)


model = Network(4*ANTENNA+1, 4*ANTENNA)
loss_fuc = MyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for j in range(MAX_EPISODES):
    indices = np.random.choice(DATA_SIZE, size=BATCH_SIZE)
    input = torch.Tensor(channel[indices,:])
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fuc(output,input)
    loss.backward()
    optimizer.step()
    print(loss.data)












