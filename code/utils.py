import subprocess
import numpy as np
import torch
import random
from score_sde_pytorch.utils import recursive_to


def random_mask_batch(batch, config):
    #
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask_min = config.model.inpainting.mask_min_len
    mask_max = config.model.inpainting.mask_max_len

    random_mask_prob = config.model.inpainting.random_mask_prob
    contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob

    lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    prob = random.random()
    if prob < random_mask_prob:
        # Random masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            rand_indices = torch.randperm(l)[:rand]

            m = torch.zeros(N)
            m[rand_indices] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    elif prob > 1-contiguous_mask_prob:
        # Contiguous masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            index = torch.randint(0, (l - rand).int(), (1,))[0]

            m = torch.zeros(N)
            m[index:index + rand] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    else:
        mask = torch.ones(B, N) # No masking

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch


def selected_mask_batch(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask = torch.zeros(B, N)

    res_mask = mask_info.split(",")
    for r in res_mask:
        if ":" in r:
            start_idx, end_idx = r.split(":")
            mask[:, int(start_idx):int(end_idx)+1] = 1
        else:
            mask[:,int(r)] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch


def get_condition_from_batch(config, batch, mask_info=None):
    batch_size = batch["coords_6d"].shape[0]
    out = {}
    for c in config.model.condition:
        if c == "length":
            lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]
            mask = torch.zeros(batch_size, config.data.max_res_num,
                               config.data.max_res_num).bool()  # B, N, N
            for idx, l in enumerate(lengths):
                mask[idx, :l, :l] = True
            out[c] = mask
        elif c == "ss":
            out[c] = batch["coords_6d"][:,4:7]
        elif c == "inpainting":
            if mask_info is not None:
                batch_masked = selected_mask_batch(batch, mask_info, config)
            else:
                batch_masked = random_mask_batch(batch, config)
            out[c] = {
                "coords_6d": batch_masked["coords_6d"],
                "mask_inpaint": batch_masked["mask_inpaint"]
            }

    return recursive_to(out, config.device)


def get_mask_all_lengths(config, batch_size=16):
    # all_lengths = np.arange(config.data.min_res_num, config.data.max_res_num+1)
    all_lengths = np.arange(5, config.data.max_res_num+1)

    mask = torch.zeros(len(all_lengths), batch_size, config.data.max_res_num,
                       config.data.max_res_num).bool()  # L, B, N, N

    for idx,l in enumerate(all_lengths):
        mask[idx, :, :l, :l] = True

    return mask


def run_tmalign(path1, path2, binary_path="tm/TMalign", fast=True):
    cmd = [binary_path, path1, path2]
    if fast:
        cmd += ["-fast"]
    result = subprocess.run(cmd, capture_output=True)
    result = result.stdout.decode("UTF-8").split("\n")
    if len(result) < 10: return 0. # when TMalign throws error
    tm = result[13].split(" ")[1].strip()
    return float(tm)

def show_all_channels(sample, path=None, nrows=1, ncols=8):
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.1,
                     share_all=True
                     )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    ax_idx = 0
    for s in sample:
        for ch in range(ncols):
            grid[ax_idx].imshow(s[ch])
            ax_idx += 1

    if path:
        plt.savefig(path)