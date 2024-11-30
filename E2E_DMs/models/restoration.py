import torch
import torch.nn as nn
import utils
import torchvision
import os
import numpy as np


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, r=None,last_TF=True,save_file_path='/fs/lustre/cita/xuduo/project_ML/DDPM_Bmag/data_orion_Turb_new/data_combine_temp/',save_file_name='test'):
    # def restore(self, val_loader, r=None,last_TF=False):
        image_folder = os.path.join(self.config.data.save_test_dir, self.config.data.dataset)
        # image_folder = os.path.join('/ADA_Storage/ddh9ms/xuduo/project_CASI/Diffusion_2D/scripts/results/test/', self.config.data.dataset)

        with torch.no_grad():
            data_save_all=[]
            for i, (x, y) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :self.config.data.channels, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r,last_TF=last_TF)
                if not last_TF:
                    x_output_temp=[]
                    for ctt_slice in range(len(x_output[0])):
                        x_output_temp.append(inverse_data_transform(x_output[0][ctt_slice]).to('cpu').numpy())
                    x_output_temp=np.asarray(x_output_temp)
                    # x_combine=np.concatenate((x.to('cpu').numpy(),x_output),1).to('cpu')
                    data_save_all.append(x_output_temp)
                else:
                    x_output = inverse_data_transform(x_output)
                    x_combine=torch.cat((x,x_output),1)
                    data_save_all.append(x_combine.numpy())
                # utils.logging.save_image(x[:, :1, :, :], os.path.join(image_folder, f"{y}_cond.png"))
                # utils.logging.save_image(x[:, 1:, :, :], os.path.join(image_folder, f"{y}_true.png"))
                # utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_pred.png"))
            data_save_all=np.asarray(data_save_all)
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            np.save(os.path.join(save_file_path, save_file_name+'.npy'),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Bmag_testB30_0207.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_T_step_plot_1.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_testonly_64_0717_noheat.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_MonR2_ds2_step_2_medpad_withnoise_0526.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_Aquila_all_image_crop_step_2_norm_1225.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_Ophiuchus_all_image_crop_step_2_norm_1225.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_Perseus_all_image_crop_step_2_norm_1225.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_MonR2_all_image_crop_step_2_norm_0822.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_test_OOD_1.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_CygnusX_all_image_crop_step_2_norm_avgpool1_1225.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_test_SFV2_1219.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_MonR2_step_2_1225.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_Ophiuchus_all_image_crop_step_2_norm.npy"),data_save_all)
            # np.save(os.path.join(image_folder, "pred_Er_Perseus_all_image_crop_step_2_norm.npy"),data_save_all)
            
    def diffusive_restoration(self, x_cond, r=None,last_TF=True):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn([x_cond.size()[0],self.config.model.out_ch,x_cond.size()[2],x_cond.size()[3]], device=self.diffusion.device)
        # torch.randn_like(x[:, self.input_chn_num:, :, :])
        # print(x_cond.shape,x.shape)
        x_output = self.diffusion.sample_image(x_cond, x, last=last_TF,patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list