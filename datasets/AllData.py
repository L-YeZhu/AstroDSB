# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB for astrophysical observational inversion
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
# from astropy.io import fits

class AllData:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.ToTensor()

    def get_loaders(self, parse_patches=False):

        

        data_all_array=np.load("PATH_TO_PREPROCESSED_ASTRO_DATA")

        X_data_all=data_all_array[0,:,:,:] # input 
        Y_data_all=data_all_array[1,:,:,:] # target

        x_train, x_test, y_train, y_test = train_test_split(X_data_all, Y_data_all, test_size=0.205,random_state=42)
        
        
        train_dataset = AllDataset(X=x_train,Y=y_train,
                                          n=8,
                                          patch_size=128,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)
        val_dataset = AllDataset(X=x_test,Y=y_test, n=4,
                                        patch_size=128,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=18, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    

        return train_loader, val_loader


class AllDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, patch_size, n, transforms, parse_patches=False,random_crop=True):
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

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n,random_crop=self.random_crop)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            target_img = self.n_random_crops(target_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]).to(torch.float32), self.transforms(target_img[i]).to(torch.float32)], dim=0)
                       for i in range(self.n)]

            return torch.stack(outputs, dim=0), img_id
        else:
            ## output shape [bs, 2, 128, 128]
            return torch.cat([self.transforms(input_img).to(torch.float32), self.transforms(target_img).to(torch.float32)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return self.total_sample
