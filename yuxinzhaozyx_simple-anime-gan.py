#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')




from torch.utils.data import Dataset
import os
from PIL import Image

class AnimateDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = [os.path.join(path, file) for file in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """ get a PIL image """
        path = self.image_paths[index]
        image = Image.open(path, "r")
        if self.transform:
            image = self.transform(image)
        return image




import torch
import torch.nn as nn 

# DCGAN
class Generator(nn.Module):
    def __init__(self, input_dim = 100):
        super().__init__()
        
        self.main = nn.Sequential(
            # input_dim x 1 x 1  to  512 x 4 x 4
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 512 x 4 x 4  to  256 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256 x 8 x 8  to  128 x 16 x 16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 16 x 16  to  64 x 32 x 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 32 x 32  to  3 x 64 x 64
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.__weight_init()

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.main(x)
        return x

    def __weight_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 3 x 64 x 64  to  64 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32  to  128 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16  to  256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8  to  512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4  to  1 x 1 x 1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.__weight_init()

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1)
        return x

    def __weight_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)




import torch

class Manager(object):
    
    def __init__(self, netG, netD=None, optimizerG=None, optimizerD=None, criterion=None, device='cpu'):
        self.netG = netG
        self.netD = netD 
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.device = device

    def train(self, dataloader, epoch):
        if not (self.netG and self.netD and self.optimizerD and self.optimizerG and self.criterion):
            raise Exception("Manager has not enough init parameters to train")

        for index, images in enumerate(dataloader):
            batch_size = images.size(0)

            """ Update D network: maximize log(D(x)) + log(1 - D(G(z))) """
            self.netD.zero_grad()

            # train with all real batch
            # real label
            labels = torch.full((batch_size,), 1, device=self.device)
            outputs = self.netD(images.to(self.device)).view(-1)
            errD_real = self.criterion(outputs, labels)
            errD_real.backward()
            D_x = outputs.mean().item()

            # train with all fake batch
            # Generate batch of latent vectors
            noises = torch.randn(batch_size, 100, device=self.device)
            # Generate fake image batch with G
            fakes = self.netG(noises)
            # fake label
            labels.fill_(0)
            outputs = self.netD(fakes).view(-1)
            errD_fake = self.criterion(outputs, labels)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = outputs.mean().item()

            # add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # update D's parameters
            self.optimizerD.step()


            """ Update G network: maximize log(D(G(z))) """
            self.netG.zero_grad()

            # real label
            labels.fill_(1)
            outputs = self.netD(fakes).view(-1)
            errG = self.criterion(outputs, labels)
            errG.backward()
            D_G_z2 = outputs.mean().item()

            # update G's parameters
            self.optimizerG.step()

            # output training status
            print('Epoch: %d [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, index + 1, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))




get_ipython().system('mkdir log')
get_ipython().system('mkdir generated')




import torch
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
import os
import time

if __name__ == '__main__':
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset_path = '../input/data/data/'
    if not os.path.exists(dataset_path):
        dataset_path = '../input/anime-faces/data/data/'
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    dataset = AnimateDataset(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    # model
    netG = Generator()
    netD = Discriminator()
    netG.to(device)
    netD.to(device)

    # optimizer
    optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    schedulerG = lr_scheduler.ExponentialLR(optimizerG, gamma=0.9)
    schedulerD = lr_scheduler.ExponentialLR(optimizerD, gamma=0.9)
    
    # criterion
    criterion = torch.nn.BCELoss()

    # manager
    manager = Manager(netG, netD, optimizerG, optimizerD, criterion, device)

    # generate
    generated_save_path = './generated'
    num_generated_image = 5

    # epochs
    epochs = 1000

    # load
    start_epoch = 1
    if os.path.exists('../input/anime-gan-weights/lastest_checkpoint.log'):
        lastest_checkpoint = torch.load('../input/anime-gan-weights/lastest_checkpoint.log')
        lastest_saved_epoch = lastest_checkpoint['epoch']
        netG.load_state_dict(torch.load('../input/anime-gan-weights/G-weight-{:0>8}.log'.format(lastest_saved_epoch)))
        netD.load_state_dict(torch.load('../input/anime-gan-weights/D-weight-{:0>8}.log'.format(lastest_saved_epoch)))
        optimizerG.load_state_dict(lastest_checkpoint['optimizerG'])
        optimizerD.load_state_dict(lastest_checkpoint['optimizerD'])
        start_epoch = lastest_saved_epoch + 1

    
    for epoch in range(start_epoch, epochs+1):
        start_time = time.time()
        
        print("\n---- EPOCH %d ----\n" % epoch)
        # train one epoch
        manager.train(dataloader, epoch)
        schedulerG.step()
        schedulerD.step()
        
        end_time = time.time()
        print("cost time: ", end_time - start_time)

        if epoch % 20 == 0:
            # save weights
            torch.save(netG.state_dict(), './log/G-weight-{:0>8}.log'.format(epoch))
            torch.save(netD.state_dict(), './log/D-weight-{:0>8}.log'.format(epoch))
            lastest_checkpoint = {'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict(), 'epoch': epoch}
            torch.save(lastest_checkpoint, './log/lastest_checkpoint.log')
        
        # save some generated image
        noises = torch.randn(num_generated_image, 100, device=device)
        with torch.no_grad():
            images = netG(noises).cpu()
            for i in range(num_generated_image):
                image = transforms.ToPILImage()(images[i])
                image.save(os.path.join(generated_save_path, '{:0>8}_{:0>4}.jpg'.format(epoch, i+1)))




# path = './generated'
# for epoch in range(start_epoch, epochs+1):
#     fig = plt.figure()
#     for i in range(num_generated_image):
#         image = np.asarray(Image.open(os.path.join(path, '{:0>8}_{:0>4}.jpg'.format(epoch, i+1))))
#         ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[], title='epoch %d'%epoch)
#         plt.imshow(image)




import zipfile
import os
path = './generated'
with zipfile.ZipFile('generated-{:0>8}-{:0>8}.zip'.format(start_epoch, epochs), 'w') as z:
    for file in os.listdir(path):
        file = os.path.join(path,file)
        z.write(file)




get_ipython().system('rm -rf ./generated')

