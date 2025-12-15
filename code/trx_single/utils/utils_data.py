import warnings

import os
import string
import json
import torch
import torch.nn as nn
import numpy as np

class bcolors:
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = '\033[96m'
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'

def get_clusters(clust_file):
    all_lst = {}
    current_lst = []
    for line in open(clust_file):
        if line.startswith('>Cluster'):
            if current_lst:
                all_lst[clstr_id] = current_lst
            clstr_id = int(line.split()[1])
            current_lst = []
        else:
            pid = line.split()[2].split('>')[1].split('.')[0]
            current_lst.append(pid)
    train_lst = [all_lst[idx] for idx in all_lst]
    return train_lst


def mymsa_to_esmmsa(msa, input_type='msa', in_torch=False):
    if in_torch:
        device = msa.device
        token = torch.tensor([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8,
                              11, 22, 19, 7, 30], device=device)
        cls = torch.zeros_like(msa[..., 0:1], device=device)
        eos = 2 * torch.ones_like(msa[..., 0:1], device=device)
        # token, cls, eos = map(lambda x: torch.from_numpy(x), [token, cls, eos])
        if input_type == 'fasta':
            return torch.cat([cls, token[msa], eos], dim=-1)
        else:
            return torch.cat([cls, token[msa]], dim=-1)
    else:
        token = np.array([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8,
                          11, 22, 19, 7, 30])
        cls = np.zeros_like(msa[..., 0:1])
        eos = 2 * np.ones_like(msa[..., 0:1])
        if input_type == 'fasta':
            return np.concatenate([cls, token[msa], eos], axis=-1)
        else:
            return np.concatenate([cls, token[msa]], axis=-1)


def esmmsa_to_mymsa(msa):
    device = msa.device
    d = {5: 0, 10: 1, 17: 2, 13: 3, 23: 4, 16: 5, 9: 6, 6: 7, 21: 8, 12: 9, 4: 10, 15: 11, 20: 12, 18: 13, 14: 14, 8: 15, 11: 16, 22: 17, 19: 18, 7: 19, 30: 20, 32: 20}
    val_if_not_shown = 20  # 0 can be any other number within dtype range
    return msa.cpu().apply_(lambda val: d.get(val, val_if_not_shown)).to(device)


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

def parse_seq(fasta,input_='file'):
    if input_ == 'file':
        seq_str = read_fasta(fasta)
    else:
        seq_str = fasta
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase + '*'))
    line = str(seq_str).rstrip().translate(table)
    seqs.append(line.rstrip().translate(table))
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    seq = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq[seq == alphabet[i]] = i

    # treat all unknown characters as gaps
    seq[seq > 20] = 20

    return seq


def save_to_json(obj, file):
    with open(file, "w") as f:
        jso = json.dumps(obj, cls=NpEncoder)
        f.write(jso)


def read_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
