'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''
from math import log10

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
    
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('tanh'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('tanh'))
        init.orthogonal_(self.conv4.weight)

criterion = nn.MSELoss()

def train(model, device, train_loader, optimizer, epoch):
    '''
    train the model
    '''
    #model.train()
    epoch_loss = 0
    print("Epoch "+str(epoch))
    for iter, batch in enumerate(train_loader,1):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()




def test(model, device, test_loader):
    '''
    test the model
    '''
    #model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for batch in test_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            mse = criterion(output, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

    print('\nTest set: Avg. PSNR: {:.4f} dB'.format(avg_psnr/len(test_loader)))

    return


# ''' image transformation for training '''
# train_transform = torchvision.transforms.Compose([
#                            torchvision.transforms.RandomAffine(5,translate=(0.1,0.1)),
#                            torchvision.transforms.ToTensor()
#                            ])

# ''' image transformation for test '''
# test_transform = torchvision.transforms.Compose([
#                            torchvision.transforms.ToTensor()
#                            ])



# ''' image transformation for image generation '''
# gen_transform = torchvision.transforms.Compose([
#                            torchvision.transforms.ToTensor()
#                            ])
