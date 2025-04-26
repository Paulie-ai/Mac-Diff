# coding=utf-8
"Get 6D features for protein conformation"
import numpy as np
import scipy.spatial
import math
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import esm
import logging
from tqdm.contrib.concurrent import process_map
from pathlib import Path

pre_trained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
pre_trained_model.eval().to('cuda:0')
batch_strs_ALL = ['LAGVSERTIDPKQNFYMHWCXBUZO.-']
batch_tokens_all = torch.tensor([[ 0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  2]])
str_tokens_pairs = {res: toks for res, toks in zip(batch_strs_ALL[0], batch_tokens_all.squeeze(0)[1:-1])}


def crop_strategy(seqs, coords_6d, max_L=256):
    """
    Current Strategys: every time random crop seqs and correspond 6D repr of one random conformations.
    Given seqs and max_L
    return crop seqs  and coords_6d
    """
    
    L = len(seqs)
    if L > max_L:
        start_idx = np.random.randint(0, L - max_L)
        last_idx = start_idx+ max_L
        crop_seqs = seqs[start_idx: start_idx+ max_L]
        coords_6d = coords_6d[:,start_idx: last_idx, start_idx: last_idx]
        return crop_seqs, coords_6d
    else:
        return seqs, coords_6d


def seqs2esm(aa_str):
    """
    Given seqs, return seq repr
    """
    toks =torch.tensor([[0]])
    for i in aa_str:
        tok = str_tokens_pairs[i].unsqueeze(0).unsqueeze(0)
        toks = torch.cat((toks, tok), dim=1)  
    toks = torch.cat((toks, torch.tensor([[2]])), dim=1).to('cuda:0') # 1,130.
    
    with torch.no_grad():
        # seq_esm_token: L+2
        results = pre_trained_model(toks,  repr_layers=[33], return_contacts=True)
        # (1, L+2, 1280)
        token_repr = results['representations'][33]
        # (1,33,20,L+2,L+2)
        # attentions = results['attentions'] # not used now
        # (1,L,L)
        contacts = results['contacts']
        
        emb_out ={
            'token_repr': token_repr.squeeze(0)[1:-1], # rm 0 dim
            'contacts': contacts.squeeze(0)
        }
        return emb_out

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


##### Functions below adapted from trRosetta https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/coords6d.py
# calculate dihedral angles defined by 4 sets of points
letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

def get_dihedrals(a, b, c, d):
    
    # Ignore divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')
    
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c
    
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)

# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, dmax=20.0, normalize=True):
    # (l,3,3)
    # (10,l,3,3)
    nres = xyz.shape[0]

    # three anchor atoms
    N  = xyz[:,0]  # (l,3)
    Ca = xyz[:,1]  # (l,3)
    C  = xyz[:,2]  # (l,3)

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca  # (l,3)

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)  # must be two dimension.
    indices = kdCb.query_ball_tree(kdCb, dmax)  # (l,) list.

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]  # (25264,)?
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres), dmax).astype(float)  # [l,l]
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    # Normalize all features to [-1,1]
    if normalize:
        # [4A, 20A]
        dist6d = (dist6d / dmax*2) - 1
        # [-pi, pi]
        omega6d = omega6d / math.pi
        # [-pi, pi]
        theta6d = theta6d / math.pi
        # [0, pi]
        phi6d = (phi6d / math.pi*2) - 1

    coords_6d = np.stack([dist6d,omega6d,theta6d,phi6d],axis=-1)

    return coords_6d


class ProteinProcessedDataset(Dataset):

    def __init__(self, root_path, min_res_num=40, max_res_num=256, images_per_epoch=10, flag=0, index=None):
        super().__init__()
        self.root_path = root_path
        # self.data_paths = os.listdir(root_path)
        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.images_per_epoch = images_per_epoch
       
        paths = [npz for npz in Path(root_path).rglob('*.npz')]
        structures = self.parse_pdb(paths)
        self.structures = structures

    def parse_pdb(self, paths):
        logging.info(f"Processing dataset of length {len(paths)}...")
        data = list(map(self.parse_npz, paths))
        split_clst_data = self.split_feats(data)

        return split_clst_data
    
    def split_feats(self, data_lists):
        combined_list = []
        for data_clust_dict in data_lists:
            if data_clust_dict is not None:
                split_dicts = [{key: data_clust_dict[key][i] for key in data_clust_dict.keys()} for i in range(len(data_clust_dict['id']))]
                combined_list.extend(split_dicts)
        return combined_list
    
    def parse_npz(self, paths):
        """
        Input: paths of npy files
        Outputs: ids, features(5,N,N) no padding to max_residue_length, token of seqs 
        """
        raw_data = np.load(paths, allow_pickle=True)

        for i in raw_data:
            datas = raw_data[i].item()
            ids = i

        fastas = datas['fasta']

        ids_list = []
        crds_list = []
        aa_list = []
        aa_str_list = []
        mask_pair_list = []
        seq_repr_list = []
        cmap_list = []

        
        for bb_coords_clust in datas['bb_coords']:
            N, L3, _ = bb_coords_clust.shape
            image_idx = np.random.randint(N)
            try:
                bb_coords = bb_coords_clust.reshape(N,L3//3,3,3)[image_idx]*10
            except ValueError as e:
                print(f'Raw data error id is {ids}')
            
            coords_6d = get_coords6d(bb_coords, dmax=20.0, normalize=True)
            coords_6d = np.nan_to_num(coords_6d)
            
            nres = coords_6d.shape[0]
            padding = np.ones((nres,nres)).reshape(nres,nres,1)
            coords_6d = np.concatenate([coords_6d, padding], axis=-1)
            coords_6d = coords_6d.transpose(2,0,1) 
            
            # Crop seqs  
            fasta, coords_6d = crop_strategy(fastas, coords_6d, max_L=self.max_res_num)

            L = coords_6d.shape[1]
            if L > self.max_res_num or L < self.min_res_num: return None

            # aa token    
            aa = [letter_to_num[i] for i in fasta] 

            mask_pair = np.ones((L, L), dtype=float)
            
            esm_out = seqs2esm(fasta)
            seq_repr, contacts_m = esm_out['token_repr'], esm_out['contacts']

            assert L == len(aa)
            # append all clust to list
            ids_list.append(ids)
            crds_list.append(torch.from_numpy(coords_6d).to(dtype=torch.float32))
            aa_list.append(torch.from_numpy(np.asarray(aa)).to(dtype=torch.long))
            aa_str_list.append(fasta)
            mask_pair_list.append(torch.from_numpy(mask_pair).to(dtype=torch.bool))
            seq_repr_list.append(seq_repr.cpu())
            cmap_list.append(contacts_m.cpu())
       
        torch.cuda.empty_cache()    

        return {
            'id': ids_list,
            'coords_6d': crds_list,
            'aa': aa_list,
            'aa_str': aa_str_list,
            'mask_pair': mask_pair_list,
            'seq_repr': seq_repr_list,
            'contacts_m': cmap_list,
        }

    def __len__(self):
        return len(self.structures) * self.images_per_epoch
    
    def __getitem__(self, idx):
        idx_r = idx // self.images_per_epoch
        return self.structures[idx_r]

    def to_tensor(self, d):
        feat_dtypes = {
            "id": None,
            "coords_6d": torch.float32,
            "aa": torch.long,
            "aa_str": None,
            "mask_pair": torch.bool,
            "seq_repr": torch.float32,
            "contacts_m": torch.float32,
        }
        
        for k,v in d.items():
            if feat_dtypes[k] is not None:
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

        return d

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
            # data_padded = {
            #     k: self._pad_last(v, max_length, value=self._get_value(k)) for k,v in data.items()
            # }
            data_padded = {}
            for k, v in data.items():
                v = self._pad_last(v, max_length, value=self._get_value(k))
                data_padded[k] = v
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)
