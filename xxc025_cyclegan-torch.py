#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install torchsummary')




import os,itertools
import numpy as np
from matplotlib import pylab as plt
from PIL import Image
import torch 
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import time
import random
from skimage import color




device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image




class Mydataset(data.Dataset):
    def __init__(self,root,transform=None):
        imgs=os.listdir(root)
        self.imgs=sorted([os.path.join(root,img) for img in imgs])
        self.transform=transform
        
    def __getitem__(self,index):
        img_path=self.imgs[index]
        image=Image.open(img_path)
        if image.mode != "RGB":
            image = to_rgb(image)
        if self.transform is not None:
            image=self.transform(image)
        return image
    
    def __len__(self):
        return len(self.imgs)




trainA_path='../input/horse2zebra/horse2zebra/trainA'
trainB_path='../input/horse2zebra/horse2zebra/trainB'
testA_path='../input/horse2zebra/horse2zebra/testA'
testB_path='../input/horse2zebra/horse2zebra/testB'

transform=transforms.Compose([
    transforms.Resize(int(256* 1.12), Image.BICUBIC),
    transforms.RandomCrop((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_A=Mydataset(trainA_path,transform=transform)
train_loader_A=data.DataLoader(train_A,batch_size=1,shuffle=True,num_workers=8)
train_B=Mydataset(trainB_path,transform=transform)
train_loader_B=data.DataLoader(train_B,batch_size=1,shuffle=True,num_workers=8)
test_A=Mydataset(testA_path,transform=transform)
test_loader_A=data.DataLoader(test_A,batch_size=1,shuffle=False)
test_B=Mydataset(testB_path,transform=transform)
test_loader_B=data.DataLoader(test_B,batch_size=1,shuffle=False)




A_data=test_A.__getitem__(11).unsqueeze(0)#(1,3,256,256)
B_data=test_B.__getitem__(81).unsqueeze(0)
fig=plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.transpose(A_data[0]*0.5+0.5,(1,2,0)))
plt.subplot(1,2,2)
plt.imshow(np.transpose(B_data[0]*0.5+0.5,(1,2,0)))




def plot_train_result(real_img,gen_img,recon_img,epoch,save=True):
    fig,axs=plt.subplots(2,3,figsize=(10,10))
    imgs=[real_img[0].cpu(),gen_img[0].cpu(),recon_img[0].cpu(),
          real_img[1].cpu(),gen_img[1].cpu(),recon_img[1].cpu()]
#     imgs=[real_img[0].data.cpu().numpy(),gen_img[0].data.cpu().numpy(),recon_img[0].data.cpu().numpy(),
#           real_img[1].data.cpu().numpy(),gen_img[1].data.cpu().numpy(),recon_img[1].data.cpu().numpy()]
    for ax,img in zip(axs.flatten(),imgs):
        ax.axis('off')
        img=img.squeeze()
        img=np.transpose((img*0.5+0.5),(1,2,0))
        ax.imshow(img)
    title='Epoch {}'.format(epoch+1)
    fig.text(0.5,0.04,title,ha='center')
    if save:
        plt.savefig('./%d.png'%(epoch+1))
    plt.show()




# plt.figure(figsize=(8,8))
# for i in range(2):
#     real_batch=next(iter(train_loader_A))
#     real_batch=(real_batch+1)/2
#     plt.subplot(1,2,i+1)
#     plt.axis('off')
#     plt.imshow(np.transpose(real_batch[0].to(device).cpu(),(1,2,0)))




class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self,pool_size):
        self.pool_size=pool_size
        if self.pool_size>0:
            self.num_imgs=0
            self.images=[]
            
    def query(self,images):
        if self.pool_size==0:
            return images
        return_images=[]
        for image in images: #考虑到训练时的batch
            image=torch.unsqueeze(image.data,0)
            if self.num_imgs<self.pool_size:
                self.num_imgs+=1
                self.images.append(image)
                return_images.append(image)
            else:
                p=random.uniform(0,1)
                if p>0.5: #use previously stored image
                    random_id=random.randint(0,self.pool_size-1)
                    tmp=self.images[random_id].clone()
                    self.images[random_id]=image
                    return_images.append(tmp)
                else: #use current image 
                    return_images.append(image)
        return_images=torch.cat(return_images,0)
        return return_images




class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,k,s,p,activation='relu',norm=True):
        super(conv_block,self).__init__()
        self.norm=norm
        self.activation=activation
        self.conv=nn.Conv2d(ch_in,ch_out,kernel_size=k,stride=s,padding=p)
        self.bn=nn.InstanceNorm2d(ch_out)
        
    def forward(self,x):
        if self.norm:
            out=self.bn(self.conv(x))
        else:
            out=self.conv(x)
        if self.activation=='relu':
            out=F.relu(out)
        elif self.activation=='lrelu':
            out=F.leaky_relu(out,0.2)
        elif self.activation=='tanh':
            out=torch.tanh(out)
        elif self.activation=='sig':
            out=torch.sigmoid(out)
        return out




class deconv_block(nn.Module):
    def __init__(self,ch_in,ch_out,k,s,p,op):
        super(deconv_block,self).__init__()
        self.deconv=nn.ConvTranspose2d(ch_in,ch_out,kernel_size=k,stride=s,padding=p,output_padding=op)
        self.bn=nn.InstanceNorm2d(ch_out)
        
    def forward(self,x):
        out=self.bn(self.deconv(x))
        out=F.relu(out)
        return out




class resnetblock(nn.Module):
    def __init__(self,ch_in,ch_out,k,s,p):
        super(resnetblock,self).__init__()
        self.res=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in,ch_out,kernel_size=k,stride=s,padding=p),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out,ch_out,kernel_size=k,stride=s,padding=p),
            nn.InstanceNorm2d(ch_out),
        )
        
    def forward(self,x):
        out=x+self.res(x)
        return out




nc=3
gf=64
df=64




class Generator(nn.Module):
    def __init__(self,num_res=6):
        super(Generator,self).__init__()
        #c7s164,d128,d256
        self.pad=nn.ReflectionPad2d(3)
        self.conv1=conv_block(nc,gf,7,1,0)
        self.conv2=conv_block(gf,gf*2,3,2,1)
        self.conv3=conv_block(gf*2,gf*4,3,2,1)
        #R256*6
        self.res_blocks=[]
        for i in range(num_res):
            self.res_blocks.append(resnetblock(gf*4,gf*4,3,1,0))
        self.res_blocks=nn.Sequential(*self.res_blocks)
        #u128,u64
        self.deconv1=deconv_block(gf*4,gf*2,3,2,1,1)
        self.deconv2=deconv_block(gf*2,gf,3,2,1,1)
        self.deconv3=conv_block(gf,nc,7,1,0,activation='tanh',norm=False)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data,0.0,0.02)
                nn.init.constant_(m.bias.data,0.0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data,1.0,0.02)
                nn.init.constant_(m.bias.data,0.0)
        
    def forward(self,x):
        enc1=self.conv1(self.pad(x))
        enc2=self.conv2(enc1)
        enc3=self.conv3(enc2)
        
        res=self.res_blocks(enc3)
        
        dec1=self.deconv1(res)
        dec2=self.deconv2(dec1)
        out=self.deconv3(self.pad(dec2))
        return out
    




# model=Generator().to(device)
# summary(model,(3,256,256))




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1=conv_block(nc,df,4,2,1,activation='lrelu',norm=False)
        self.conv2=conv_block(df,df*2,4,2,1,activation='lrelu')
        self.conv3=conv_block(df*2,df*4,4,2,1,activation='lrelu')
        self.conv4=conv_block(df*4,df*8,4,1,1,activation='lrelu')
        self.conv5=conv_block(df*8,1,4,1,1,norm=False)
        self.conv_blocks=nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data,0.0,0.02)
                nn.init.constant_(m.bias.data,0.0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data,1.0,0.02)
                nn.init.constant_(m.bias.data,0.0)
                
    def forward(self,x):
        out=self.conv_blocks(x)
        return out          




# model=Discriminator().to(device)
# summary(model,(3,256,256))




G_AB=Generator().to(device)
G_BA=Generator().to(device)
D_A=Discriminator().to(device)
D_B=Discriminator().to(device)




lr_G=0.0002
lr_D=0.0002
G_optimizer=torch.optim.Adam(itertools.chain(G_AB.parameters(),G_BA.parameters()),lr=lr_G,betas=(0.5,0.999))
#G_BA_optimizer=torch.optim.Adam(G_BA.parameters(),lr=lr_G,betas=(0.5,0.999))
D_A_optimizer=torch.optim.Adam(D_A.parameters(),lr=lr_D,betas=(0.5,0.999))
D_B_optimizer=torch.optim.Adam(D_B.parameters(),lr=lr_D,betas=(0.5,0.999))




MSE_Loss=nn.MSELoss()
L1_Loss=nn.L1Loss()
G_AB_avg_losses=[]
G_BA_avg_losses=[]
D_A_avg_losses=[]
D_B_avg_losses=[]
cycle_A_avg_losses=[]
cycle_B_avg_losses=[]
iters=0




# Generated image pool
num_pool = 50
fake_A_pool = ImagePool(num_pool)
fake_B_pool = ImagePool(num_pool)




def sample_images():
    G_AB.eval()
    G_BA.eval()
    real_A=A_data.to(device)
    fake_B=G_AB(real_A).detach()
    recon_A=G_BA(fake_B).detach()
    
    real_B=B_data.to(device)
    fake_A=G_BA(real_B).detach()
    recon_B=G_AB(fake_A).detach()
    
    plot_train_result([real_A,real_B],[fake_B,fake_A],[recon_A,recon_B],
                           epoch,save=True)




for epoch in range(60):
    start=time.time()
    G_AB_losses=[]
    G_BA_losses=[]
    D_A_losses=[]
    D_B_losses=[]
    cycle_A_losses=[]
    cycle_B_losses=[]
    if (epoch+1)>50:
        G_optimizer.param_groups[0]['lr']-=lr_G/(60-50)
        #G_BA_optimizer.param_groups[0]['lr']-=lr_G/(200-100)
        D_A_optimizer.param_groups[0]['lr']-=lr_D/(60-50)
        D_B_optimizer.param_groups[0]['lr']-=lr_D/(60-50)
    
    for i,(real_A,real_B) in enumerate(zip(train_loader_A,train_loader_B)):
        real_A=real_A.to(device)
        real_B=real_B.to(device)
        #####################################
        #           Train G
        #####################################
        G_optimizer.zero_grad()
        #A-->B
        fake_B=G_AB(real_A)
        D_B_fake_decision=D_B(fake_B)
        G_AB_loss=MSE_Loss(D_B_fake_decision,torch.full(D_B_fake_decision.size(),1,device=device))
        recon_A=G_BA(fake_B)
        cycle_A_loss=L1_Loss(recon_A,real_A)*10
        
        id_A=G_BA(real_A)
        id_A_loss=L1_Loss(id_A,real_A)*10*0.5
        #B-->A
        fake_A=G_BA(real_B)
        D_A_fake_decision=D_A(fake_A)
        G_BA_loss=MSE_Loss(D_A_fake_decision,torch.full(D_A_fake_decision.size(),1,device=device))
        
        recon_B=G_AB(fake_A)
        cycle_B_loss=L1_Loss(recon_B,real_B)*10
        
        id_B=G_AB(real_B)
        id_B_loss=L1_Loss(id_B,real_B)*10*0.5
        #update
        G_loss=G_AB_loss+G_BA_loss+cycle_A_loss+cycle_B_loss+id_A_loss+id_B_loss
        
        G_loss.backward()
        G_optimizer.step()
        
        #####################################
        #           Train D_A
        #####################################
        D_A_optimizer.zero_grad()
        D_A_real_decision=D_A(real_A)
        D_A_real_loss=MSE_Loss(D_A_real_decision,torch.full(D_A_real_decision.size(),1,device=device))
        
        fake_A = fake_A_pool.query(fake_A)
        
        D_A_fake_decision=D_A(fake_A.detach())
        D_A_fake_loss=MSE_Loss(D_A_fake_decision,torch.full(D_A_fake_decision.size(),0,device=device))
        #update
        D_A_loss=(D_A_real_loss+D_A_fake_loss)*0.5
        
        D_A_loss.backward()
        D_A_optimizer.step()
        
        #####################################
        #           Train D_B
        #####################################
        D_B_optimizer.zero_grad()
        D_B_real_decision=D_B(real_B)
        D_B_real_loss=MSE_Loss(D_B_real_decision,torch.full(D_B_real_decision.size(),1,device=device))
        
        fake_B = fake_B_pool.query(fake_B)
        
        D_B_fake_decision=D_B(fake_B.detach())
        D_B_fake_loss=MSE_Loss(D_B_fake_decision,torch.full(D_B_fake_decision.size(),0,device=device))
        #update
        D_B_loss=(D_B_real_loss+D_B_fake_loss)*0.5
        
        D_B_loss.backward()
        D_B_optimizer.step()
        
        #iters' loss
        G_AB_losses.append(G_AB_loss.item())
        G_BA_losses.append(G_BA_loss.item())
        D_A_losses.append(D_A_loss.item())
        D_B_losses.append(D_B_loss.item())
        cycle_A_losses.append(cycle_A_loss.item())
        cycle_B_losses.append(cycle_B_loss.item())
        
        if (i%100==0):
            print('Epoch [%d/%d],[%d/%d],D_A_loss:%.4f,D_B_loss:%.4f,G_AB_loss:%.4f,G_BA_loss:%.4f'
             % (epoch+1,60,i,len(train_loader_A),D_A_loss.item(),D_B_loss.item(),G_AB_loss.item(),G_BA_loss.item()))
    
    print('Time for one epoch is {} sec'.format(time.time()-start))
    #each epoch loss    
    G_AB_avg_loss=torch.mean(torch.FloatTensor(G_AB_losses))
    G_BA_avg_loss=torch.mean(torch.FloatTensor(G_BA_losses))
    D_A_avg_loss=torch.mean(torch.FloatTensor(D_A_losses))
    D_B_avg_loss=torch.mean(torch.FloatTensor(D_B_losses))
    cycle_A_avg_loss=torch.mean(torch.FloatTensor(cycle_A_losses))
    cycle_B_avg_loss=torch.mean(torch.FloatTensor(cycle_B_losses))
    #all epochs loss list
    G_AB_avg_losses.append(G_AB_avg_loss.item())
    G_BA_avg_losses.append(G_BA_avg_loss.item())
    D_A_avg_losses.append(D_A_avg_loss.item())
    D_B_avg_losses.append(D_B_avg_loss.item())
    cycle_A_avg_losses.append(cycle_A_avg_loss.item())
    cycle_B_avg_losses.append(cycle_B_avg_loss.item())
    
    sample_images()
    #test result
#     with torch.no_grad():
#         test_real_A=test_real_A_data.to(device)
#         test_fake_B=G_AB(test_real_A).detach()
#         test_recon_A=G_BA(test_fake_B).detach()
    
#         test_real_B=test_real_B_data.to(device)
#         test_fake_A=G_BA(test_real_B).detach()
#         test_recon_B=G_AB(test_fake_A).detach()
    
#         plot_train_result([test_real_A,test_real_B],[test_fake_B,test_fake_A],[test_recon_A,test_recon_B],
#                            epoch,save=True)




plt.figure(figsize=(10,5))
plt.title('G and D Loss during Training')
plt.plot(G_AB_avg_losses,label='G_AB')
plt.plot(G_BA_avg_losses,label='G_BA')
plt.plot(D_A_avg_losses,label='D_A')
plt.plot(D_B_avg_losses,label='D_B')
plt.plot(cycle_A_avg_losses,label='cycle_A')
plt.plot(cycle_B_avg_losses,label='cycle_B')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




get_ipython().system('nvidia-smi')

