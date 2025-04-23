# coding=utf-8
"Get 6D features for protein conformation"
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import esm
from pathlib import Path

class ProteinProcessedDataset(Dataset):

    def __init__(self, root_path, min_res_num=40, max_res_num=256, images_per_epoch=10, index=None):
        super().__init__()
        self.root_path = root_path
        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.images_per_epoch = images_per_epoch

        if index is None:
            pt_file = [pts for pts in Path(root_path).rglob('*.pt')]
            ####
            datas_list = []
            print("loading test datas...")
            for pt in pt_file:
                data = torch.load(pt)
                datas_list += data
            self.structures = datas_list
            print(f"load all test datas.")
        else:          
            print(f"loading index: {index} ...")
            pt_file = os.path.join(root_path, '%d_128.pt'%index)
            self.structures = torch.load(pt_file)    # 1000* [protein_file.length//10]
            
            print(f"load index:{index} finished.")

    def __len__(self):
        return len(self.structures) 
    
    def __getitem__(self, idx):
        return self.structures[idx]
    
class PDBProteinProcessedDataset(Dataset):

    def __init__(self, root_path, min_res_num=40, max_res_num=256, images_per_epoch=10, index=None):
        super().__init__()
        self.root_path = root_path
        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.images_per_epoch = images_per_epoch

        pt_file = [pts for pts in Path(root_path).rglob('*.pt')]
        ####
        datas_list = []
        print("loading test datas...")
        for pt in pt_file:
            data = torch.load(pt)
            datas_list.append(data)  # [{},{}...{}]
        self.structures = datas_list
        print(f"load all PDB datas.")

    def __len__(self):
        return len(self.structures) 
    
    def __getitem__(self, idx):
        return self.structures[idx]

class PaddingCollate(object):

    def __init__(self, max_len=None):
        super().__init__()
        self.max_len = max_len

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x

            # Pairwise embeddings TODO: not very elegant
            if len(x.shape) >= 2 and x.shape[-1] != 3 and x.shape[-1] == x.shape[-2]:
                x = F.pad(x, (0,n-x.shape[-1],0,n-x.shape[-2]), value=value)
                return x

            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, str):
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_value(k):
        if k in ["aa_str"]:
            return "_"
        elif k == "aa":
            return 21 # masking value
        elif k == "id":
            return ''
        else:
            return 0

    def __call__(self, data_list):
        max_length = self.max_len if self.max_len else max([len(data["aa"]) for data in data_list])
        data_list_padded = []
        for data in data_list:
            data_padded = {}
            for k, v in data.items():
                v = self._pad_last(v, max_length, value=self._get_value(k))
                data_padded[k] = v
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)