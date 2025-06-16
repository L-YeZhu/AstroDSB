# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB for astrophysical observational inversion, for submission review only
# --------------------------------------------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.utils as tvu

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
# from .network import Image256Net
from .network import Density128Net
from .network import ISFR64Net
from .network import MAG128Net
from .diffusion import Diffusion

from ipdb import set_trace as debug
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


## this if for Taurus B128
def crop_image(img, patch_size=128, step=8):

    """Crop an image into overlapping 128x128 patches with a step size of 8."""

    h, w = img.shape[:2]
    patches = []
    positions = []
    print("check h and w:", h, w)

    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            positions.append((x,y))
    return patches, positions, (h, w)


def merge_patches(patches, positions, img_size, patch_size=128, step=8):

    """Merge overlapping 128x128 patches back into an image of size img_size."""
    h, w = img_size
    merged_img = np.zeros((h,w), dtype=np.float32)
    # Assuming 3 channels
    count_map = np.zeros((h,w), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        merged_img[y:y+patch_size, x:x+patch_size] += patch
        count_map[y:y+patch_size,x:x+patch_size] += 1


    # Avoid division by zero

    count_map[count_map == 0] = 1

    merged_img /= count_map 
    # Normalize by the number of overlapping patches


    # Crop the overshoot at the boundary

    merged_img = merged_img[:h, :w]

    return merged_img


def save_checkpoint(state, filename):
    # print("check path:",os.path.dirname(filename))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        self.net = Density128Net(log, noise_levels=noise_levels, cond=True)
        # self.net = ISFR64Net(log, noise_levels=noise_levels, cond=True)
        # self.net = MAG128Net(log, noise_levels=noise_levels, cond=True)

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")



        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond



    def train(self, opt, train_loader, val_loader):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        self.it = 0

        net.train()    

        # n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for epoch in range(500):
            print('epoch:', epoch)
            for i, (x, y, z) in enumerate(train_loader):
                # ===== sample boundary pair =====
                self.it += 1

                #### this is for originally density task
                x = x.to(opt.device)
                x = data_transform(x)
                x0 = x[:,1,:,:].unsqueeze(1)
                x1 = x[:,0,:,:].unsqueeze(1)
                cond_x1 = x1
                x1 = x1 + 0.1 * torch.randn_like(x1)



                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond_x1) # or cond = x1

                #### need to recalculate the pred
                pred_avg = torch.mean(pred, 1, True)

                loss = F.mse_loss(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update()
                if sched is not None: sched.step()


                if self.it % 10 == 0:
                    print(f"iteration: {self.it}, loss: {loss.item()}")
                    # self.writer.add_scalar(self.it, 'loss', loss.detach())
                    writer.add_scalar("Loss/train", loss, self.it)

                if self.it == 1 or self.it % 1000 == 0:
                    # if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt") 

                        net.eval()
                        self.evaluation(opt, self.it, val_loader)
                        net.train()

                    if opt.distributed:
                        torch.distributed.barrier()
                    print("checkpoint saved at iteration:", self.it)                 

                if self.it == 1 or self.it % 1000 == 0: # 0, 0.5k, 3k, 6k 9k
                    net.eval()
                    self.evaluation(opt, self.it, val_loader)
                    net.train()

                    if opt.distributed:
                        torch.distributed.barrier()
                    print("checkpoint saved at iteration:", self.it) 

        writer.close()



    def eval(self, opt, val_loader):
        self.writer = util.build_log_writer(opt)
        log = self.log
        mask = None


        checkpoint = torch.load('./PATH_TO_MODEL_CHECKPOINT')

        self.net.load_state_dict(checkpoint['net'])
        log.info(f"[Net] Loaded network ckpt!")
        self.ema.load_state_dict(checkpoint["ema"])
        log.info(f"[Ema] Loaded ema ckpt!")
        print("Finish loading")

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        self.it = 0

        net.eval()    

        image_folder = "./results/psb_taurus"


        ################ taurus inference ############
        x_input = np.load("./data/taurus_L1495_column_density_map_rot_norm_128.npy")
        patches, positions, img_size = crop_image(x_input) ## [18, 128, 128]
        print("check patches:", np.shape(patches))
        patches = np.array(patches)
        cond = torch.from_numpy(patches).unsqueeze(1).float().to(opt.device)
        preds = []
        # results
        for i in range(cond.size()[0]):
            img_corrupt = cond[i,:,:,:].unsqueeze(0)
            img_corrupt = data_transform(img_corrupt.to(opt.device))
            x1 = img_corrupt
            cond_x1 = x1
            x1 = x1 + 0.1 * torch.randn_like(x1)
            print("check size:", img_corrupt.size(), x1.size())
            # exit()
            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond_x1, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
            )
            img_recon = xs[:, 0, ...]
            img_recon = inverse_data_transform(img_recon)
            pred_tmp = img_recon.squeeze().detach().cpu().numpy()
        


            preds.append(pred_tmp)
        print("check preds size:", np.shape(preds))
        reconstructed_img = merge_patches(preds, positions, img_size)
        print("check reconstructed_img:", np.shape(reconstructed_img))
        np.save("recons_taurus_inverse.npy", reconstructed_img)


        ################ OOD inference ############


        x_input = np.load("/OOD_FILE")
        cond = x_input["X_train"]
        cond = torch.from_numpy(cond).unsqueeze(1).float().to(opt.device) #.to(self.device).float()
        target = x_input["Y_train"]
        target = torch.from_numpy(target).unsqueeze(1).float().to(opt.device) #.to(self.device).float()

        for i in range(cond.size()[0]):
            img_corrupt = cond[i,:,:,:] #.unsqueeze(0)
            img_corrupt = data_transform(img_corrupt.to(opt.device))
            img_clean = target[i,:,:,:] #.unsqueeze(0)
            img_clean = data_transform(img_clean.to(opt.device))


            x1 = img_corrupt
            cond_x1 = x1
            x1 = x1 + 0.1 * torch.randn_like(x1)



            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond_x1, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
            )


            img_recon = xs[:, 0, ...]
            img_clean = inverse_data_transform(img_clean)
            img_corrupt = inverse_data_transform(img_corrupt)
            img_recon = inverse_data_transform(img_recon)
            img_recon_avg = img_recon.mean(dim=1, keepdim=True)

            output_num = torch.cat((img_corrupt, img_recon_avg.to(opt.device), img_clean), dim=1)
            if i == 0:
                results_target = output_num
            else:
                results_target = torch.cat((results_target, output_num), dim=0)
            results_store = results_target.detach().cpu().numpy()


        writer.close()


    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0


    @torch.no_grad()
    def evaluation(self, opt, it, val_loader):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")
        cond = None
        mask = None
        image_folder = "./name_of_folder"

        for i, (x, y, z) in enumerate(val_loader):
            n = x.size(0)
            img_corrupt = data_transform(x.to(opt.device))
            img_clean = data_transform(y.to(opt.device))
            x1 = img_corrupt
            cond_x1 = x1
            x1 = x1 + 0.1 * torch.randn_like(x1)
       
            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond_x1, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
            )            

            img_recon = xs[:, 0, ...]
            img_recon = torch.mean(img_recon, 1, True)

            img_clean = inverse_data_transform(img_clean)
            img_corrupt = inverse_data_transform(img_corrupt)
            img_recon_inverse= inverse_data_transform(img_recon)

            for j in range(n):
                save_image(img_clean[j], os.path.join(image_folder, str(it), f"{i+j}_gt.png"))
                save_image(img_corrupt[j], os.path.join(image_folder, str(it), f"{i+j}_cond.png"))
                save_image(img_recon[j], os.path.join(image_folder, str(it), f"{i+j}_pred.png"))
                save_image(img_recon_inverse[j], os.path.join(image_folder, str(it), f"{i+j}_pred_inv.png"))

            
            break
        
        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()

