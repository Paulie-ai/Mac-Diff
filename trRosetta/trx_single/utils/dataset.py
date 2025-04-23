import zipfile

import zlib

import random

import warnings

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join, isfile, isdir
from os import listdir

from trx_single.utils.utils_data import mymsa_to_esmmsa


class SingleDataset(Dataset):
    def __init__(self,
                 targets,
                 root_dir="/home/wangwk/db/train_onlyseq",
                 lengthmax=300,
                 labels=['dist', 'omega', 'theta', 'phi'],
                 nbins={'dist': 37, 'omega': 25, 'theta': 25, 'phi': 13},
                 clusters=None,
                 return_msa=False,
                 warning=False
                 ):
        super(SingleDataset, self).__init__()

        self.datadir = root_dir
        self.lengthmax = lengthmax
        self.labels = labels
        self.nbins = nbins
        self.clusters = clusters
        self.return_msa = return_msa
        self.warning = warning

        files = []
        if clusters is None:
            for p in targets:
                if p=='long_list_all':continue
                sample_file = join(self.datadir, p + '.npz')
                if not isfile(sample_file):
                    if warning:
                        warn = f'{sample_file} missed!'
                        warnings.warn(warn)
                    continue
                files.append(sample_file)
        else:
            for clstr_id in targets:
                files_clstr = []
                for p in clusters[clstr_id]:
                    sample_file = join(self.datadir, p + '.npz')
                    if not isfile(sample_file):
                        if warning:
                            warn = f'{sample_file} missed!'
                            warnings.warn(warn)
                        continue
                    files_clstr.append(sample_file)
                files.append(files_clstr)

        #  files list
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # file name
        file = self.files[idx]
        if isinstance(file, list):
            if file:
                file = random.choice(file)
            else:
                if self.warning:
                    warn = f'no vaild npz file for cluster {idx}!'
                    warnings.warn(warn)
                return {}

        # data
        try:
            sample = dict(np.load(file))
        except (zipfile.BadZipFile, zlib.error, ValueError):
            if self.warning:
                warn = f'npz file {file} is broken!'
                warnings.warn(warn)
            return {}
        if 'seq' not in sample:
            sample['seq'] = sample['msa'][0]
        L = sample['seq'].shape[-1]

        # crop
        if L > self.lengthmax:
            crop_size = self.lengthmax // 2 - 15
            left = max(0, int(10 ** np.random.uniform(np.log10(L / 2 - crop_size)) - 10))
            right = L - max(int(10 ** np.random.uniform(np.log10(L / 2 - crop_size)) - 10), 0)
            res_id = list(range(left, left + crop_size)) + list(range(right - crop_size, right))

            sample['seq'] = sample['seq'][res_id]
            if self.return_msa and 'msa' in sample:
                sample['msa'] = sample['msa'][:, res_id]

            for lab in self.labels:
                if lab in ['theta','phi'] and 'theta_asym' in sample:
                    sample[lab] = sample[lab + '_asym'][res_id][:, res_id]
                else:
                    sample[lab] = sample[lab][res_id][:, res_id]
        else:
            res_id = range(L)
            for lab in self.labels:
                if lab in ['theta','phi'] and 'theta_asym' in sample:
                    sample[lab] = sample[lab + '_asym']

        sample['seq_esm'] = mymsa_to_esmmsa(sample['seq'], input_type='fasta')
        sample['idx'] = np.array(res_id).astype(np.int32)

        mask = self.nan_mask(sample)
        return self.onehot(self.process_nan(self.binning(sample), mask))

    def binning(self, sample):

        bins = np.linspace(2, 20, self.nbins['dist'])
        bins180 = np.linspace(0.0, np.pi, self.nbins['phi'])
        bins360 = np.linspace(-np.pi, np.pi, self.nbins['omega'])

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

    def onehot(self, binned_sample):
        for lab in self.labels:
            nb = self.nbins[lab]
            binned_sample[lab] = (np.arange(nb) == binned_sample[lab][..., None])
        return binned_sample

    def nan_mask(self, sample):
        mask = {}
        for lab in self.labels:
            mask[lab] = np.isnan(sample[lab])
        return mask

    def process_nan(self, sample, mask):
        for lab in self.labels:
            sample[lab][mask[lab]] = -1
        return sample
