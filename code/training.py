from pathlib import Path

import score_sde_pytorch.losses as lossss
import argparse
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
from score_sde_pytorch.models import utils as mutils
import torch
from torch.utils import tensorboard
from score_sde_pytorch.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to
from dataset_process import ProteinProcessedDataset, PaddingCollate, PDBProteinProcessedDataset
import yaml
from easydict import EasyDict
import time
from utils import random_mask_batch
import shutil
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debugs', type=bool, default=None)
    parser.add_argument('--pt', type=str, default=None)
    args = parser.parse_args()
    ###
    local_rank = 0
    accumulation_steps = 2  ## accumulate.

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    device = config.device
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if args.debugs:
        processed_dataset_path = '/mnt/disk/datas_protein/test_pt'
        test_dataset_path = '/mnt/disk/datas_protein/test_pt'
    else:
        processed_dataset_path = config.data.dataset_path
        test_dataset_path = config.data.dataset_testpath
    
    print("----------------finished------------------------")

    # Create directories for experimental logs
    if args.resume is not None:
        workdir = Path(args.resume)
    else:
        workdir = Path("training", Path(args.config).stem, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        workdir.mkdir(exist_ok=True,parents=True)
        # Save config to workdir
        shutil.copy(args.config, workdir.joinpath("config.yml"))

    sample_dir = workdir.joinpath("samples")
    sample_dir.mkdir(exist_ok=True)

    tb_dir = workdir.joinpath("tensorboard")
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = get_model(config, local_rank)
    # print(score_model)
    
    print('Loaded UNetModel!')
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = lossss.get_optimizer(config, score_model.parameters())

    state = dict(optimizer=optimizer, model=score_model, llm=None, ema=ema, step=0, epoch=0)
    # Create checkpoints directory
    checkpoint_dir = workdir.joinpath("checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "best_eval.pth")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_meta_dir.parent.mkdir(exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if checkpoint_meta_dir.is_file():
        state = restore_checkpoint(checkpoint_meta_dir, state, device)
        initial_epoch = int(state['epoch'])+1
        initial_dl = int(state['dl'])
        initial_sub = int(state['sub'])
    else:
        if args.pt is not None:  # used for loading checkpoint.
            pt_dir = args.pt
            state = restore_checkpoint(pt_dir, state, device, config.model.init_self) 
            initial_epoch = int(state['epoch'])+1
            initial_dl = int(state['dl'])
            initial_sub = int(state['sub'])
        else:
            initial_epoch = 0
            initial_dl = 0
            initial_sub = 0

    if initial_sub != 0:
        initial_epoch = initial_epoch-1   

    print(f"Starting from epoch {initial_epoch}, total steps {initial_dl}...")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")


    # Build training and evaluation functions
    optimize_fn = lossss.optimization_manager(config)
    eval_step_fn = lossss.get_step_fn(sde, train=False, optimize_fn=optimize_fn)

    min_avg_eval_loss = 1e10
    eval_last_save_epoch = -1
    all_step_one_epoch = 0

    for epoch in range(initial_epoch, config.training.epochs):
        all_eval_losses = []
        train_file_num = 0
        train_now_sub = 0 
        if initial_sub != 0 and epoch == initial_epoch: 
            train_file_num = initial_dl
            train_now_sub = initial_sub
        all_train_losses = []
        for i in range(train_now_sub, 10):
            train_ds = ProteinProcessedDataset(processed_dataset_path, max_res_num=config.data.max_res_num, images_per_epoch=config.data.images_per_epoch, index=i)
            train_dl = torch.utils.data.DataLoader(
                train_ds,
                shuffle=True,
                batch_size=config.training.batch_size,
                collate_fn=PaddingCollate(config.data.max_res_num)
            )

            train_progress_bar = tqdm(train_dl)
            train_batch_num = len(train_dl)
            train_last_save_epoch = -1
            accumulation_loss = 0

            for step, batch in enumerate(train_progress_bar):
                batch = recursive_to(batch, device)
                batch = random_mask_batch(batch, config)
                
                coords_6d = batch["coords_6d"]
                mask_pair = batch["mask_pair"]
               
                caption_emb = batch['seq_repr'].to(coords_6d.device)
                esm_contact = batch['contacts_m'].to(coords_6d.device)
                
                score_fn = mutils.get_score_fn(sde, state['model'], train=True)
                t = torch.rand(coords_6d.shape[0], device=coords_6d.device) * (sde.T - 1e-5) + 1e-5
                z = torch.randn_like(coords_6d)
                mean, std = sde.marginal_prob(coords_6d, t)
                perturbed_data = mean + std[:, None, None, None] * z

                conditional_mask = torch.ones_like(coords_6d).bool()

                mask = mask_pair.unsqueeze(1) * conditional_mask
                num_elem = mask.reshape(mask.shape[0], -1).sum(dim=-1)

                perturbed_data = torch.where(mask, perturbed_data, coords_6d)
                score = score_fn(perturbed_data, t, caption_emb, esm_contact)
                losses = torch.square(score * std[:, None, None, None] + z) * mask 
                losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
                losses = losses / (num_elem + 1e-8)
                loss = torch.mean(losses)
                loss.backward()

                if( (step+1) % accumulation_steps)==0: 
                    update_step = step // accumulation_steps
                    # print(f'now step is:{step+1}, update_step is:{update_step}')
                    accumulation_loss += loss.item()
                    if config.optim.warmup > 0:
                        for g in state['optimizer'].param_groups:
                            g['lr'] = config.optim.lr * np.minimum(state['step']/ config.optim.warmup, 1.0)
                            if epoch >= 2:
                                g['lr'] = config.optim.lr * 0.1
                    
                    if config.optim.grad_clip >= 0:
                        torch.nn.utils.clip_grad_norm_(state['model'].parameters(), max_norm=config.optim.grad_clip)
                    state['optimizer'].step()
                    state['optimizer'].zero_grad()
                    state['step'] += 1
                    state['ema'].update(state['model'].named_parameters())

                    all_train_losses.append(accumulation_loss)
                    avg_loss = sum(all_train_losses) / len(all_train_losses)
                    train_progress_bar.set_description(f"Epoch: {epoch}, File:{i}, Step: {step + 1}/{train_batch_num}, batch_loss: {accumulation_loss}, avg_loss: {avg_loss}")

                    accumulation_loss = 0 
                    cur_step =  initial_epoch * initial_dl + (epoch-initial_epoch) * all_step_one_epoch + train_file_num + update_step

                    if cur_step % config.training.log_freq == 0:
                        writer.add_scalar("training_loss", loss, cur_step)
            
            ### saving this file checkpoint.
            train_file_num = train_file_num + train_batch_num//accumulation_steps
            checkpoint_sub_dir = checkpoint_dir.joinpath('Epoch_%d_length_128'%epoch)
            checkpoint_sub_dir.mkdir(exist_ok=True)
            if i == 9:
                all_step_one_epoch = train_file_num
                state['dl'] = train_file_num
                state['sub'] = 0
                state['epoch'] = epoch 
                save_checkpoint(checkpoint_dir.joinpath('epoch_%d.pth'%(epoch)), state)
            else:
                state['dl'] = train_file_num 
                state['sub'] = i+1
                state['epoch'] = epoch
                save_checkpoint(checkpoint_sub_dir.joinpath('epoch_%d_SubEpoch_%d.pth'%(epoch, i)), state)
            del train_ds.structures
            if epoch == config.training.epochs-1 and (i >= 4 and i < 9):
                test_datasets = ProteinProcessedDataset(test_dataset_path, max_res_num=config.data.max_res_num,  images_per_epoch=config.data.images_per_epoch)
                test_ds = test_datasets
                test_dl = torch.utils.data.DataLoader(
                        test_ds,
                        shuffle=False,
                        batch_size=config.training.batch_size,
                        collate_fn=PaddingCollate(config.data.max_res_num)
                    )
                eval_progress_bar = tqdm(test_dl)

                for eval_step, eval_batch in enumerate(eval_progress_bar):
                    eval_batch = recursive_to(eval_batch, device)
                    eval_batch = random_mask_batch(eval_batch, config)
                    eval_loss = eval_step_fn(state, eval_batch, condition=config.model.condition)
                    all_eval_losses.append(eval_loss.item())

                if len(all_eval_losses) > 0: # same as train part 
                        avg_eval_loss = sum(all_eval_losses) / len(all_eval_losses)

                        writer.add_scalar("avg_eval_loss", avg_eval_loss, epoch)

                        if avg_eval_loss < min_avg_eval_loss:
                            min_avg_eval_loss = avg_eval_loss
                            print(f'Eval: Update best eval model at epoch {epoch}, eval_avg_loss:', avg_eval_loss)
                            save_checkpoint(checkpoint_meta_dir, state)
                            eval_last_save_epoch = epoch
                            update_eval_best = True
                        else:
                            print(f'Eval: Not update best model at epoch {epoch}, cur_avg_loss:', avg_eval_loss,
                                    'min_avg_loss:', min_avg_eval_loss,
                                    'seen at epoch:', eval_last_save_epoch)

        # -----------------------------Evaluation---------------------------------
        # Generate and save sample
        test_datasets = ProteinProcessedDataset(test_dataset_path, max_res_num=config.data.max_res_num,  images_per_epoch=config.data.images_per_epoch)
        test_ds = test_datasets
        test_dl = torch.utils.data.DataLoader(
                test_ds,
                shuffle=False,
                batch_size=config.training.batch_size,
                collate_fn=PaddingCollate(config.data.max_res_num)
            )
        eval_progress_bar = tqdm(test_dl)

        for eval_step, eval_batch in enumerate(eval_progress_bar):
            eval_batch = recursive_to(eval_batch, device)
            eval_batch = random_mask_batch(eval_batch, config)
            eval_loss = eval_step_fn(state, eval_batch, condition=config.model.condition)
            all_eval_losses.append(eval_loss.item())

        if len(all_eval_losses) > 0:
                avg_eval_loss = sum(all_eval_losses) / len(all_eval_losses)

                writer.add_scalar("avg_eval_loss", avg_eval_loss, epoch)

                if avg_eval_loss < min_avg_eval_loss:
                    min_avg_eval_loss = avg_eval_loss
                    print(f'Eval: Update best eval model at epoch {epoch}, eval_avg_loss:', avg_eval_loss)
                    save_checkpoint(checkpoint_meta_dir, state)
                    eval_last_save_epoch = epoch
                    update_eval_best = True
                else:
                    print(f'Eval: Not update best model at epoch {epoch}, cur_avg_loss:', avg_eval_loss,
                            'min_avg_loss:', min_avg_eval_loss,
                            'seen at epoch:', eval_last_save_epoch)
                    
    print(f"Finally, best eval(minimum eval loss) epoch:{eval_last_save_epoch}")
        

if __name__ == "__main__":
    main()
