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

class AllData:
    def __init__(self, config):
        self.config = config
        # self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.transforms = torchvision.transforms.ToTensor()

    def get_loaders(self, parse_patches=False):

        data_path=self.config.data.data_dir
        data_filename=self.config.data.dataset
        
        # data_obs=fits.open('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy/herschel_data/C_2014_06_norm_zoom1.fits')[0].data
        # data_obs=fits.open('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy/herschel_data/H_2014_06_norm.fits')[0].data

        # data_obs=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy_arepo/Ndens_turb_11_crop_grav_norm.npy')
        # data_obs_Bmag=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy_arepo/Bmag_turb_11_crop_grav_norm.npy')
        # data_obs=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy_arepo/data_arepo_turb_sn50_crop.npy')
        # data_obs_Bmag=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_temp_npy_arepo/data_arepo_turb_sn50_crop.npy')
        

        # data_obs=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_gizmo/Prob_TurbGrav_mu4_normX.npy')  ###_alpha2
        # data_obs_Bmag=np.load('/scratch/users/xuduo117/project_CASI/Diffusion_Bmag/data_gizmo/Prob_TurbGrav_mu4_normY.npy')

        # X_data_all=data_obs###+0.2 ##0.03697479160272606
        # mask_density_train=X_data_all<(1./3.)
        # Y_data_all=data_obs_Bmag
        # Y_data_all[mask_density_train]=0.0
        
        # X_data_all=data_obs[np.newaxis,:,:]

        # Y_data_all=data_obs[np.newaxis,:,:]*0.0+0.0
        # X_data_all=data_obs
        # Y_data_all=data_obs*0.0+0.0

        # x_train=X_data_all
        # y_train=Y_data_all
        # X_sample_step_1=X_data_all
        # Y_sample_step_1=Y_data_all

        # data_all_array=np.load(data_path+data_filename+'.npz')
        data_all_array=np.load(data_path+data_filename+'.npy')
        # data_all_array_1=np.load(data_path+data_filename)
        # # data_all_array_2=np.load(data_path+'test_B_mag_0828_TestBNC_cutden1e21.npy')
        # data_all_array_2=np.load(data_path+'test_B_mag_0828_TestB30_cutden1e21.npy')
        
        # data_all_array=np.load(data_path+'train_Bpos_enzo_4chan_all.npz')
        
        
        
        # # X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test
        # x_train=data_all_array['X_train'][:,0:4,:,:]
        
        # # x_train[:,0,:,:]=x_train[:,0,:,:]-np.log10(2)/6
        # # mask_temp=x_train[:,0,:,:]<0
        # # x_train[:,0,:,:][mask_temp]=0
        
        # y_train=data_all_array['Y_train']
        # # x_test=data_all_array['X_test']
        # # y_test=data_all_array['Y_test']
        # x_test=x_train
        # y_test=y_train
        
        x_train=data_all_array
        y_train=data_all_array[:,0:1,:,:]*0.0
        x_test=x_train
        y_test=y_train
        
        
        
        
        
        
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
        

        # x_train=data_all_array_1[0,:,:,:]
        # mask_density_train=x_train<(1./3.)
        # y_train=data_all_array_1[1,:,:,:]
        # y_train[mask_density_train]=0.0
        
        # x_test=data_all_array_2[0,:,:,:]
        # mask_density_test=x_test<(1./3.)
        # y_test=data_all_array_2[1,:,:,:]
        # y_test[mask_density_test]=0.0
        
        # X_data_all=data_all_array[0,:,:,:]
        # Y_data_all=data_all_array[1,:,:,:]
        
        # x_train=X_data_all
        # y_train=Y_data_all
        # X_sample_step_1=X_data_all
        # Y_sample_step_1=Y_data_all

        # X_sample_step_1=X_data_all[878:879]   ##1418   ##1895  878
        # Y_sample_step_1=Y_data_all[878:879]   ##1418   ##1895
        ######
        # X_data_all=X_data_all[:,np.newaxis,:,:]
        # Y_data_all=Y_data_all[:,np.newaxis,:,:]
        ######
        # x_train, x_test, y_train, y_test = train_test_split(X_data_all, Y_data_all, test_size=0.205,random_state=42)
        # ### x_train, x_test, y_train, y_test = train_test_split(X_data_all, Y_data_all, test_size=0.005,random_state=42)
        

        
        train_dataset = AllDataset(X=x_train,Y=y_train,
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          parse_patches=parse_patches)
        val_dataset = AllDataset(X=x_test,Y=y_test, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches)

        # train_dataset = AllDataset(X=X_sample_step_1,Y=Y_sample_step_1, n=self.config.training.patch_n,
        #                                 patch_size=self.config.data.image_size,
        #                                 transforms=self.transforms,
        #                                 parse_patches=parse_patches)
        # val_dataset = AllDataset(X=X_sample_step_1,Y=Y_sample_step_1, n=self.config.training.patch_n,
        #                                 patch_size=self.config.data.image_size,
        #                                 transforms=self.transforms,
        #                                 parse_patches=parse_patches)


        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)) # create your datset
        # val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)) # create your datset
        

        # my_dataloader = torch.utils.data.DataLoader(my_dataset) # create your dataloader

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
        #                                            shuffle=True, num_workers=self.config.data.num_workers,
        #                                            pin_memory=True)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
        #                                          shuffle=False, num_workers=self.config.data.num_workers,
        #                                          pin_memory=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size)
    

        return train_loader, val_loader
        # return train_dataset, val_dataset



class AllDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y, patch_size, n, transforms, parse_patches=True,random_crop=True):
        super().__init__()

        self.dir = dir
        # train_list = os.path.join(dir, filelist)
        # with open(train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        # self.input_names = input_names
        # self.gt_names = gt_names
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
            # new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            new_crop = img[y[i]:y[i]+w, x[i]:x[i]+h]
            # print(new_crop.shape)
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        # input_name = self.input_names[index]
        # gt_name = self.gt_names[index]
        img_id = int(index)
        input_img=self.X_data[index]
        target_img=self.Y_data[index]

        # input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        
        # try:
        #     target_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        # except:
        #     target_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
        #         PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n,random_crop=self.random_crop)
            # print(len(input_img),input_img.shape,'before')            
            # print(len(input_img),input_img.shape)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            target_img = self.n_random_crops(target_img, i, j, h, w)
            # print(len(input_img),input_img[0].shape)
            # print(len(target_img),target_img[0].shape)
            # outputs = [torch.cat([torch.unsqueeze(self.transforms(input_img[i]),dim=0), self.transforms(target_img[i])], dim=0)
            outputs = [torch.cat([self.transforms(input_img[i]).to(torch.float32), self.transforms(target_img[i]).to(torch.float32)], dim=0)
                       for i in range(self.n)]
            # print(len(outputs),len(outputs[0]),len(outputs[0][0]),len(outputs[0][0][0]))

            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_raw, ht_raw = input_img.shape[1:3]
            # wd_new, ht_new = input_img.shape
            max_pixel_num= wd_raw ###128 ##256
            # if ht_new >= wd_new and ht_new >= max_pixel_num:
            #     wd_new = int(np.ceil(wd_new * max_pixel_num / ht_new))
            #     ht_new = max_pixel_num
            # elif ht_new <= wd_new and wd_new >= max_pixel_num:
            #     ht_new = int(np.ceil(ht_new * max_pixel_num / wd_new))
            #     wd_new = max_pixel_num
            # wd_new = int(16 * np.ceil(wd_new / 16.0))
            # ht_new = int(16 * np.ceil(ht_new / 16.0))
            # # input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # # target_img = target_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # zoomin_factor=max_pixel_num/wd_raw
            # zoomin_factor=1  ##/4.0 ##/2.0
            # input_img=ndimage.zoom(input_img,zoomin_factor)
            # target_img=ndimage.zoom(target_img,zoomin_factor)
            
            # print(input_img.shape)
            # print(target_img.shape)
            # print((self.transforms(input_img).to(torch.float32)).shape)
            
            # print(torch.cat([self.transforms(input_img).to(torch.float32), self.transforms(target_img).to(torch.float32)], dim=0)[None,].shape)

            return torch.cat([self.transforms(input_img).to(torch.float32), self.transforms(target_img).to(torch.float32)], dim=0)[None,], img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        # return len(self.input_names)
        return self.total_sample