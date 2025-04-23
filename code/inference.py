import torch
import torch.nn.functional as F
from pathlib import Path
from score_sde_pytorch.utils import get_model
import score_sde_pytorch.sde_lib as sde_lib
import score_sde_pytorch.sampling as sampling
import argparse
import yaml
from easydict import EasyDict
from utils import get_mask_all_lengths
from dataset import seqs2esm
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def padding_esm_repr(esm_repr, max_L):
    """
    Given esm_repr as seq_repr and contact_map
    Return padding data
    """
    if esm_repr.shape[0] == esm_repr.shape[1]:
        esm_padding = F.pad(esm_repr, (0, max(0, max_L - esm_repr.shape[1]), 0, max(0, max_L - esm_repr.shape[0])), mode='constant', value=0)
    elif esm_repr.shape[0] < esm_repr.shape[1]:
        esm_padding = F.pad(esm_repr, (0, 0, 0, max(0, max_L - esm_repr.shape[0])), mode='constant', value=0)
    return esm_padding

def read_fasta(file):
    fasta = ""
    with open(file, "r") as f:
        for line in f:
            if (line[0] == ">"):
                if len(fasta)>0:
                    warnings.warn('Submitted protein contained multiple chains. Only the first protein chain will be used')
                    break
                continue
            else:
                line = line.rstrip()
                fasta = fasta + line
    return fasta


def main():
    
    parser = argparse.ArgumentParser(usage = "Protein conformational ensembles generating")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file')
    parser.add_argument('--fasta', type=str, required=True,help="input sequence")
    parser.add_argument('--output', type = str, help = 'The directory where the output results are stored(*.npy).')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_samples_eval', type=int, default=20)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    batch_size = args.batch_size
    workdir = Path(args.output,"Sampling")
    print(f'Saved 6D rst in {workdir}')
    workdir.mkdir(parents=True, exist_ok=True)

    # Initialize model.
    score_model = get_model(config)
    state = dict(model=score_model, step=0, epoch=0)
    loaded_state = torch.load(args.checkpoint, map_location=config.device)
    state['model'].load_state_dict(loaded_state['model'], strict=False)

    # Load SDE
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5 
    elif config.training.sde == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    # Sampling function
    sampling_shape = (batch_size, config.data.num_channels,
                      config.data.max_res_num, config.data.max_res_num)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps) 

    print('Start sampling...')
    fasta_test = [read_fasta(args.fasta)] 

    for i in fasta_test:
        tmp = seqs2esm(i)
        context = tmp['token_repr']
        esm_contact = tmp['contacts'].to(config.device)

        mask = get_mask_all_lengths(config, batch_size=batch_size)[len(fasta_test[0])-5]
        condition = {"length":mask.to(config.device)}

        assert np.sqrt(mask[0].sum()) == len(fasta_test[0])

        context = padding_esm_repr(context, config.data.max_res_num).unsqueeze(0).expand(batch_size, -1, -1).to(config.device)
        esm_contact = padding_esm_repr(esm_contact, config.data.max_res_num).unsqueeze(0).expand(batch_size, -1, -1).to(config.device)

        for num in range(args.num_samples_eval // args.batch_size):
            generated_samples = []
            
            sample, _ = sampling_fn(state["model"], condition=condition, context=context, esm_contact=esm_contact, infer=True)

            generated_samples.append(sample.cpu())
            generated_samples = torch.cat(generated_samples, 0)

            # save the result.
            for j in range(batch_size):
                np.save(workdir.joinpath(f"sampled_{num*batch_size+j}"),generated_samples[j].unsqueeze(0).numpy())


if __name__ == "__main__":
    main()
