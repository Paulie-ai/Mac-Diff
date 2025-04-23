import warnings

import numpy as np
import os, sys
import scipy.spatial
import concurrent.futures

from Bio import PDB
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter

sys.path.insert(0, str(Path(os.path.abspath(os.path.dirname(__file__))).parent.parent))
from trx_single.utils.utils_data import parse_seq, read_fasta

retain_all_res = True

AA3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'PHD': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F', 'PRO': 'P',
          'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)


def get_neighbors(residues, dmax):
    nres = len(residues)

    # three anchor atoms
    N = np.stack([np.array(residues[i]['N'].coord) for i in range(nres)])
    Ca = np.stack([np.array(residues[i]['CA'].coord) for i in range(nres)])
    C = np.stack([np.array(residues[i]['C'].coord) for i in range(nres)])

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    for i in range(nres):
        resname = residues[i].resname
        resname = resname.strip()
        if (resname != "GLY"):
            Cb[i] = np.array(residues[i]['CB'].coord)

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d


def parse_pdb_6d(pid, pdb_file, save_pth, fasta_file=None):
    # load PDB
    pp = PDB.PDBParser()
    structure = pp.get_structure('myStructureName', pdb_file)[0]
    chain = structure.child_list[0]
    if retain_all_res:
        residues = [res for res in chain.child_list if PDB.is_aa(res)]
    else:
        residues = [res for res in chain.child_list if PDB.is_aa(res) and 'CB' in res and 'N' in res]

    seq_pdb = ''.join([AA3to1[res.resname.strip()] for res in residues])
    if fasta_file is None:
        seq = seq_pdb
    else:
        seq = read_fasta(fasta_file)
        if seq != seq_pdb:
            warnings.warn(f'fasta and pdb not match for {pid}!')
    seq_array = parse_seq(seq, input_='str')[0]

    # 6D coordinates
    dist, omega, theta_asym, phi_asym = get_neighbors(residues, 20)

    labels = {
        'seq': seq_array,
        'dist': dist, 'omega': omega,
        'theta': theta_asym, 'phi': phi_asym
    }
    np.savez_compressed(f'{save_pth}/{pid}.npz', **labels)


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare npz files for training set', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-pdb', '--pdb_pth', type=str, required=True, help='Location of folder storing PDB files of training set')
    parser.add_argument('-o', '--out_pth', type=str, required=True, help='Location of folder to save npz files to')
    parser.add_argument('-fa', '--fasta_pth', type=str, default=None,
                        help='Location of folder storing FASTA files of training set\nIf not provided, the sequence extracted from PDB files will be used (default: None_')
    parser.add_argument('-cpu', '--n_cpu', type=int, default=2, help='num of CPU cores to use')
    args = parser.parse_args()

    pid_lst = [f.split('.')[0] for f in os.listdir(args.pdb_pth)]
    os.makedirs(args.out_pth, exist_ok=True)

    if args.fasta_pth is None:
        pid_args = {pid: (pid, f'{args.pdb_pth}/{pid}.pdb', args.out_pth, None) for pid in pid_lst}
    else:
        pid_args = {pid: (pid, f'{args.pdb_pth}/{pid}.pdb', args.out_pth, f'{args.fasta_pth}/{pid}.fasta') for pid in pid_lst}

    executor = concurrent.futures.ProcessPoolExecutor(args.n_cpu)
    futures = [executor.submit(parse_pdb_6d, *pid_args[pid]) for pid in pid_lst]
    results = concurrent.futures.wait(futures)
