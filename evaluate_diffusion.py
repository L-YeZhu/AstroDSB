
import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration



def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/configs/surfden_1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/save_model/ckpts/test_Bmag_ddim_0207_all_ddm.pth.tar', type=str,
    # parser.add_argument('--resume', default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/save_model/ckpts/test_Bmag_ddim_0222_3chan_all_ddm.pth.tar', type=str,
    # parser.add_argument('--resume', default='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/save_model/ckpts/test_Bmag_ddim_0222_1chan_all_ddm.pth.tar', type=str,
                            help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
    # parser.add_argument("--grid_r", type=int, default=64,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
    # parser.add_argument("--sampling_timesteps", type=int, default=50,
    # parser.add_argument("--sampling_timesteps", type=int, default=1000,
                        help="Number of implicit sampling steps")
    # parser.add_argument("--test_set", type=str, default='raindrop',
    #                     help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config




def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    
    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    
    # config.data.data_dir='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/data_orion_Turb_new/data_combine_temp/'
    # config.data.data_dir='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/data_orion_outflow/'
    config.data.data_dir='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/data_obs_paul/'
    
    # for ctt_file in [0,1,2,3,4,5,6]:
    # for ctt_file in ['orion_2p5_168_log']:
    # for ctt_file in ['orion_5_168_log']:
    # for ctt_file in ['orion_outflow_2p5_168']:
    # for ctt_file in ['orion_outflow_5_168']:
    for ctt_file in ['cloud_C_4chan_norm']:
        # config.data.dataset='test_crop_'+str(ctt_file)
        config.data.dataset=str(ctt_file)
        # save_file_name='pred_'+config.data.dataset+'_mod2'+'_3chan'
        save_file_name='pred_'+config.data.dataset##+'_3chan'
        DATASET = datasets.AllData(config)
        _, val_loader = DATASET.get_loaders(parse_patches=False)
        # create model
        print("=> creating denoising-diffusion model with wrapper...")
        diffusion = DenoisingDiffusion(args, config)
        model = DiffusiveRestoration(diffusion, args, config)
        save_file_path=config.data.data_dir
        model.restore(val_loader, r=args.grid_r,save_file_path=save_file_path,save_file_name=save_file_name)



    # DATASET = datasets.AllData(config)
    # _, val_loader = DATASET.get_loaders(parse_patches=False)

    # # create model
    # print("=> creating denoising-diffusion model with wrapper...")
    # diffusion = DenoisingDiffusion(args, config)
    # model = DiffusiveRestoration(diffusion, args, config)
    # model.restore(val_loader, r=args.grid_r)
    
    
    
    

if __name__ == "__main__":
    main()