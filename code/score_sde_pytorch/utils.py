from score_sde_pytorch.models import ncsnpp
import torch
import math
from torch.nn.parallel import DistributedDataParallel as DDP

from score_sde_pytorch.models.attention import init_

def get_model(config, local_rank=None):
    score_model = ncsnpp.UNetModel(config)
    score_model = score_model.cuda()
    score_model = torch.nn.DataParallel(score_model)
    
    return score_model

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['epoch'] = loaded_state['epoch']
    state['dl'] = loaded_state['dl']
    state['sub'] = loaded_state['sub']
    return state

def save_checkpoint(ckpt_dir, state):
    saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'], 
    'epoch': state['epoch'], 
    'dl': state['dl'], 
    'sub': state['sub']
    }
    torch.save(saved_state, ckpt_dir)

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    else:
        return obj
