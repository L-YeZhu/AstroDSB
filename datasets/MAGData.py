# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB for astrophysical observational inversion, for submission review only
# --------------------------------------------------------------------------------------------------

import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from scipy import ndimage
from sklearn.model_selection import train_test_split
from astropy.io import fits

class MAGData:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.ToTensor()

    def get_loaders(self, parse_patches=False):


        data_all_array = np.load("PATH_TO_MAG_DATA")
        

        x_train=data_all_array['X_train'][:,0:4,:,:]
        
        
        y_train=data_all_array['Y_train']
        x_test=data_all_array['X_test']
        y_test=data_all_array['Y_test']
     
        
        x_train=np.moveaxis(x_train,(1,2,3),(3,1,2))
        y_train=np.moveaxis(y_train,(1,2,3),(3,1,2))
        x_test=np.moveaxis(x_test,(1,2,3),(3,1,2))
        y_test=np.moveaxis(y_test,(1,2,3),(3,1,2))
        
        x_train[np.isnan(x_train)]=0
        y_train[np.isnan(y_train)]=0
        x_test[np.isnan(x_test)]=0
        x_test[np.isnan(x_test)]=0
        
        x_train[np.isinf(x_train)]=0
        y_train[np.isinf(y_train)]=0
        x_test[np.isinf(x_test)]=0
        x_test[np.isinf(x_test)]=0
        
        

        
        train_dataset = MAGDataset(X=x_train,Y=y_train,
                                          n=8,
                                          patch_size=64,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)

        val_dataset = MAGDataset(X=x_test,Y=y_test, n=4,
                                        patch_size=64,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    

        return train_loader, val_loader



class MAGDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y, patch_size, n, transforms, parse_patches=True,random_crop=True):
        super().__init__()

        self.dir = dir

        self.X_data=X
        self.Y_data=Y
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.total_sample=X.shape[0]
        if not random_crop:
            ctt_x=np.int32((X.shape[-1])/patch_size)+1
            ctt_y=np.int32((X.shape[-2])/patch_size)+1            
            self.n = ctt_x*ctt_y
        self.parse_patches = parse_patches
        self.random_crop=random_crop

    @staticmethod
    def get_params(img, output_size, n, random_crop=True):
        w, h = img.shape##[1:3]
        # print(w,h)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        if random_crop:
            i_list = [random.randint(0, h - th) for _ in range(n)]
            j_list = [random.randint(0, w - tw) for _ in range(n)]
        else:
            ctt_x=(w)/tw
            ctt_y=(h)/th
            x_1d=np.append(np.arange(np.int32(ctt_x))*th,h-th)
            y_1d=np.append(np.arange(np.int32(ctt_y))*tw,w-tw)
            xx,yy=np.meshgrid(x_1d,y_1d)
            i_list=xx.flatten().tolist()
            j_list=yy.flatten().tolist()
            
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img[y[i]:y[i]+w, x[i]:x[i]+h]
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        img_id = int(index)
        input_img=self.X_data[index]
        target_img=self.Y_data[index]

        x = self.transforms(input_img).to(torch.float32)
        y = self.transforms(target_img).to(torch.float32)

        return x, y, img_id

    
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return self.total_sample