# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from score_sde_pytorch.models import utils as mutils
# import biotite.structure as struc
import random
from torch.cuda.amp import autocast
import torch.cuda.amp as amp
scaler = amp.GradScaler()


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  elif config.optim.optimizer == 'AdamW':
    optimizer = optim.AdamW(params, lr=config.optim.lr,  betas=(config.optim.beta1, 0.999),  eps=config.optim.eps,
                            weight_decay=config.optim.weight_decay)  # add.
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr, epoch=None,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""

    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)

    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

# Randomly dropout block adjacencies
def block_dropout(coords_6d, ss_indices, block_dropout=0.2):
  for idx in range(len(ss_indices)):
    if ss_indices[idx] == '': continue
    ss_idx = ss_indices[idx].split(",")
    indices_for_dropout = [b for b in ss_idx if random.random() < block_dropout]
    for i in indices_for_dropout:
      start, end = [int(x) for x in i.split(":")]
      coords_6d[idx,4:7, :, start:end] = 0
      coords_6d[idx,4:7, start:end, :] = 0

  return coords_6d

def get_sde_loss_fn(sde, train, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """

  def loss_fn(model, batch, condition=None, llm_components=None):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
      llm: A large language model to encode raw text.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    coords_6d = batch["coords_6d"]
    mask_pair = batch["mask_pair"]
    ### add one_hot ###
    # one_hot_aa = batch['one_hot']
    # one_hot_aa = one_hot_aa.to(coords_6d.device)
  
    caption_emb = batch['seq_repr'].to(coords_6d.device)
    esm_contact = batch['contacts_m'].to(coords_6d.device)

    # if "ss" in condition:
    #   coords_6d = block_dropout(coords_6d, batch["ss_indices"]) # Dropout block adjacencies
    score_fn = mutils.get_score_fn(sde, model, train=train)
  
    t = torch.rand(coords_6d.shape[0], device=coords_6d.device) * (sde.T - eps) + eps
  
    z = torch.randn_like(coords_6d)
    mean, std = sde.marginal_prob(coords_6d, t)
    perturbed_data = mean + std[:, None, None, None] * z

    conditional_mask = torch.ones_like(coords_6d).bool()
   
    mask = mask_pair.unsqueeze(1) * conditional_mask
    num_elem = mask.reshape(mask.shape[0], -1).sum(dim=-1)

    perturbed_data = torch.where(mask, perturbed_data, coords_6d)
    score = score_fn(perturbed_data, t, caption_emb, esm_contact)
    losses = torch.square(score * std[:, None, None, None] + z) * mask 
     # from algorithm inference. 
    losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
    losses = losses / (num_elem + 1e-8)
    loss = torch.mean(losses)

    return loss

  return loss_fn

def get_step_fn(sde, train, optimize_fn=None):
  """Create a one-step training/evaluation function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_sde_loss_fn(sde, train)

  def step_fn(state, batch, condition=None, epoch=None):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    tokenizer, llm = None, None
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch, condition=condition, llm_components=(tokenizer, llm))
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'], epoch=epoch)
      state['step'] += 1
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch, condition=condition, llm_components=(tokenizer, llm))
        ema.restore(model.parameters())

    return loss

  return step_fn