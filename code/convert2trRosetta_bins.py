"""Convert sampling 6D feat to dist and angle bins following trRosetta protocol
Gaussian dist to stablelize min, test sigma=1, without norm show better performance
"""

import os
from pathlib import Path
import pickle as pkl
import numpy as np
import math
import torch
import argparse


# Ref TrRosetta Energy Minimization with Predicted Restraints part
## Convert to nbins, then discrete to values
nbins={'dist': 37, 'omega': 25, 'theta': 25, 'phi': 13}
labels=['dist', 'omega', 'theta', 'phi']

def binning(sample):

    bins = np.linspace(2, 20, nbins['dist'])
    bins180 = np.linspace(0.0, np.pi, nbins['phi'])
    bins360 = np.linspace(-np.pi, np.pi, nbins['omega'])

    # bin distance
    sample['dist'] = np.digitize(sample['dist'], bins).astype(np.uint8)
    sample['dist'][sample['dist'] > 36] = 0

    # bin phi
    sample['phi'] = np.digitize(sample['phi'], bins180).astype(np.uint8)
    sample['phi'][sample['dist'] == 0] = 0

    # bin omega & theta
    for dihe in ['omega', 'theta']:
        sample[dihe] = np.digitize(sample[dihe], bins360).astype(np.uint8)
        sample[dihe][sample['dist'] == 0] = 0
    return sample

def onehot(binned_sample):
    for lab in labels:
        nb = nbins[lab]
        binned_sample[lab] = (np.arange(nb) == binned_sample[lab][..., None])
    return binned_sample


def main():
    parser = argparse.ArgumentParser(usage = " Convert backbone geometories value to probability distribution ")
    parser.add_argument('--sample_dir', type=str, help='inference samples with 6d')
    
    parser.add_argument("--out_ProbD", type=str,required=True, help="saved npz dir")
    parser.add_argument('--tags', type=str, default='test', help='tags like pdbid same to save dir')
    parser.add_argument('--symm', type=bool, default=True, help='symm dist and omega values')
    parser.add_argument('--sigma', type=float, default=1, help='gaussian distribution sigma, treated not single value but distribution')
    args = parser.parse_args()


    coords_path = args.sample_dir   
    coords_path = Path(coords_path)
    sampled_6d_paths = os.listdir(coords_path)
    out_path = args.out_ProbD
    #workdir = Path('sampling','toRosetta', args.tags)
    workdir = Path(out_path,'toRosetta')
    print(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for ith, path in enumerate(sampled_6d_paths):
        coords_6d_path = coords_path.joinpath(path) 

        # with open(coords_6d_path, 'rb') as f:
        coords_6d = np.load(coords_6d_path)
        if len(coords_6d.shape) == 4: # Shape is (1,l,l)
            coords_6d = coords_6d[0]

        msk = np.round(coords_6d[-1]) 
        L = math.sqrt(len(msk[msk == 1]))
        if not (L).is_integer():
            raise ValueError("Terminated due to improper masking channel...")
        else:
            L = int(L)
            print(f'Seqs {ith} length is', L)

        npz = {}
        for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
            npz[name] = np.clip(coords_6d[idx][msk == 1].reshape(L, L), -1, 1)

        # Inverse scaling
        npz["dist_abs"] = (npz["dist"] + 1) * 10 # Dist in dig is equal to 0, and if dist large than 19.9, set dist to 0 "cb_pwd many equal or colser to 19.9"
        npz["omega_abs"] = npz["omega"] * math.pi
        npz["theta_abs"] = npz["theta"] * math.pi # unsym
        npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2 # unsym

        sym = args.symm ## TODO: make input feature sym.  should use cb_pwd = cb_pwd.fill_diagonal_(0) set diag 0
        if sym: 
            npz["dist_abs"] = (npz["dist_abs"] + npz["dist_abs"].T) /2
            npz["omega_abs"] = (npz["omega_abs"] + npz["omega_abs"].T) /2
        
        # print(npz['dist_abs'])

        npz_data = dict(npz)
        samples = {}
        for k,v in npz_data.items():
            if k == 'dist_abs':
                samples['dist'] = v
                # print(k, v)
            elif k == 'phi_abs':
                samples['phi'] = v
            elif k == 'omega_abs':
                samples['omega'] = v
            elif k == 'theta_abs':
                samples['theta'] =v

        # mask = nan_mask(copy_sample)
        # bin_data = onehot(process_nan(binning(samples), mask))
        bin_data = onehot(binning(samples))

        save_data = {}
        if args.sigma:
            sigma = args.sigma
            for k, v in bin_data.items():
                save_data[k] = np.zeros_like(v, dtype=float)
                channels = nbins[k]
                x= np.arange(channels)
                
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        true_idx = np.where(v[i,j])[0]
                        if len(true_idx)>0:
                            mu = true_idx
                        else:
                            mu = x[-1]

                        gauss_data = np.exp(-np.power(x- mu, 2) / (2* np.power(sigma, 2)))
                        save_data[k][i, j, :] = gauss_data  

        npz_file = workdir.joinpath((f'sample2rosetta_{ith}'))
        if os.path.dirname(npz_file):
            os.makedirs(os.path.dirname(npz_file), exist_ok=True)

        np.savez_compressed(npz_file, **save_data)



if __name__ == "__main__":
    main()