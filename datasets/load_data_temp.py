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

class AllData:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):

        data_path=self.config.data.data_dir
        data_filename=self.config.data.dataset

        data_all_array=np.load(data_path+data_filename)

        X_data_all=data_all_array[0,:,:,:,:]
        Y_data_all=data_all_array[1,:,:,:,:]

        x_train, x_test, y_train, y_test = train_test_split(X_data_all, Y_data_all, test_size=0.205,random_state=43)


        train_dataset = AllDataset(x_train,y_train,
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)
        val_dataset = AllDataset(x_test,y_test, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllDataset(torch.utils.data.Dataset):
    def __init__(self, cube_data_X,cube_data_Y, patch_size, n, transforms, parse_patches=True,random_crop=True):
        super().__init__()

        self.dir = dir
        # train_list = os.path.join(dir, filelist)
        # with open(train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        # self.input_names = input_names
        # self.gt_names = gt_names
        self.all_data=cube_data
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        if not random_crop:
            ctt_x=np.int32((cube_data.shape[-3])/patch_size)+1
            ctt_y=np.int32((cube_data.shape[-2])/patch_size)+1            
            self.n = ctt_x*ctt_y
        self.parse_patches = parse_patches
        self.random_crop=random_crop

    @staticmethod
    def get_params(img, output_size, n, random_crop=True):
        w, h = img.shape
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
            # new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            new_crop = img[y[i]:y[i]+w, x[i]:x[i]+h]
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        # input_name = self.input_names[index]
        # gt_name = self.gt_names[index]
        img_id = np.int128(index)
        input_img=self.all_data[index,0]
        target_img=self.all_data[index,1]

        # input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        
        # try:
        #     target_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        # except:
        #     target_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
        #         PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n,random_crop=self.random_crop)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            target_img = self.n_random_crops(target_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(target_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_raw, ht_raw = input_img.shape
            wd_new, ht_new = input_img.shape
            max_pixel_num=256
            if ht_new >= wd_new and ht_new >= max_pixel_num:
                wd_new = int(np.ceil(wd_new * max_pixel_num / ht_new))
                ht_new = max_pixel_num
            elif ht_new <= wd_new and wd_new >= max_pixel_num:
                ht_new = int(np.ceil(ht_new * max_pixel_num / wd_new))
                wd_new = max_pixel_num
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            # input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # target_img = target_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            input_img=ndimage.zoom(input_img,wd_new/wd_raw)
            target_img=ndimage.zoom(target_img,wd_new/wd_raw)

            return torch.cat([self.transforms(input_img), self.transforms(target_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
