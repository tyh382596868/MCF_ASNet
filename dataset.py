import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import copy
from source_target_transforms import *
from library import my_save2image

class mydataset(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,speckle,Pha,transform=None):

        self.speckle = (speckle)      
        self.Pha = (Pha)
        self.transform = transform

        
    def __getitem__(self,idx):

        self.speckle_copy = copy.deepcopy(self.speckle)
        self.Pha_copy = copy.deepcopy(self.Pha)
        # print(self.speckle_copy.shape,self.Pha_copy.shape)
        
        if self.transform is not None:
            
            self.speckle_copy = Image.fromarray(self.speckle_copy)
            self.Pha_copy = Image.fromarray(self.Pha_copy)  
            # print(self.Amp_copy.shape,self.Pha_copy.shape)      
                
            self.speckle_copy, self.Pha_copy = self.transform((self.speckle_copy, self.Pha_copy))
            # print(self.speckle_copy.shape,self.Pha_copy.shape)
        
        return self.speckle_copy,self.Pha_copy #使数据类型和模型参数类型一样

    def __len__(self):

        return 1 
    
import matplotlib.pyplot as plt



if __name__ == '__main__':

    amp = np.loadtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/pi/amp/baseline/phase_diff_prop001.txt',dtype=np.float32,delimiter=',')
    pha = np.loadtxt('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/pi/pha/baseline/phase_diff.txt',dtype=np.float32,delimiter=',')

    transform = transforms.Compose([
            RandomResizeFromSequence([[192,256],[192*4,256*4],[192*6,256*6],[192*8,256*8],[192*10,256*10],[192*5,256*5]]),
            RandomRotationFromSequence((360)),#[0, 90, 180, 270]
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()])

    dataset = mydataset(amp,pha,transform=transform)

    # Amp,Pha = dataset[0]
    # my_save2image(Amp[0,:,:].numpy(),Pha[0,:,:].numpy(),'./combined_image.png', cmap='viridis')
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    
    epochs = 10
    for epoch in range(epochs):
        for i, (Amp,Pha) in enumerate(dataloader):
            print(Amp.shape,Pha.shape)

            my_save2image(Amp[0,0,:,:].numpy(),Pha[0,0,:,:].numpy(),f'./{epoch}_combined_image.png', cmap='viridis')
            
            
            
            


