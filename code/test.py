import sys
import os
import json
sys.path.append(os.path.abspath("score_sde_pytorch/models"))

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
import math
import tempfile
import shutil
import numpy as np
from time import time

from trx_single.utils.arguments import *
from trx_single.utils.utils_data import *
from trx_single.utils.utils_ros import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.core.scoring import atom_pair_constraint, dihedral_constraint, angle_constraint, pro_close

from pyrosetta import *
import proteinsgm_rosetta_min.run as rosettas

nbins={'dist': 37, 'omega': 25, 'theta': 25, 'phi': 13}
labels=['dist', 'omega', 'theta', 'phi']


def proteinsgm_main(list_datas, args):
    parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=str)
    # parser.add_argument('--tag', type=str, default="")
    # parser.add_argument('--index', type=int, default=1) # 1-indexing
    # parser.add_argument('--pdb', type=str, default=None)
    # parser.add_argument("--fasta", dest="FASTA",type=str, default=None)
    
    # parser.add_argument('--n_iter', type=int, default=10)
    # parser.add_argument('--dist_std', type=float, default=2) # 2
    # parser.add_argument('--angle_std', type=float, default=20) # 20
    # parser.add_argument('--fastdesign', type=bool, default=False) 
    # parser.add_argument('--fastrelax', type=bool, default=True)
    
    # parser.add_argument('--out', type=str, default=None)
    # args = parser.parse_args()

    n_iter = 10
    dist_std = 2
    angle_std = 20
    fastdesign = False
    fastrelax = True


    for num, data in enumerate(list_datas):
    ### HARD-CODED FOR PROPER NAMING ### .parent.stem
        # outPath = Path("sampling", "rosetta", args.tag,
        #            f"{Path(args.data).parent.stem}_index_{args.index}")
    
        samples = torch.from_numpy(data)
        sample = samples[0]

        msk = np.round(sample[-1])
        L = math.sqrt(len(msk[msk == 1]))
        if not (L).is_integer():
            raise ValueError("Terminated due to improper masking channel...")
        else:
            L = int(L)


        seq = read_fasta(args.fasta)
        pose = None # not used


        npz = {}
        for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
            npz[name] = np.clip(sample[idx][msk == 1].reshape(L, L), -1, 1)

        # Inverse scaling
        npz["dist_abs"] = (npz["dist"] + 1) * 10
        npz["omega_abs"] = npz["omega"] * math.pi
        npz["theta_abs"] = npz["theta"] * math.pi
        npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2

        rosettas.init_pyrosetta()

        # 10 cycles
        best_e = float("inf")
        best_pose = None
        for n in range(n_iter):
            
            # outPath_run = outPath.joinpath(f"round_{n + 1}")
            # if outPath_run.joinpath("final_structure.pdb").is_file():
            #     continue
            tmp_pose = rosettas.run_minimization(
                npz,
                seq,
                pose=pose,
                scriptdir=Path("proteinsgm_rosetta_min"),
                outPath=None,
                angle_std=angle_std,  # Angular harmonic std
                dist_std=dist_std,  # Distance harmonic std
                use_fastdesign=False,
                use_fastrelax=fastrelax,
            )

            score_fn = create_score_function("ref2015").score
            
            e = score_fn(tmp_pose)

            if e < best_e:
                best_e = e
                best_pose = tmp_pose
        
        destination_path = Path(args.output)
        # best_pose.dump_pdb(str(destination_path))
        best_pose.dump_pdb(args.output + f'/{num}' + '.pdb')
        print(f"Best structure saved to {destination_path}")


def logo():
    print('*********************************************************************')
    print('\
*           _        ____                _   _                      *\n\
*          | |_ _ __|  _ \ ___  ___  ___| |_| |_ __ _               *\n\
*          | __| \'__| |_) / _ \/ __|/ _ \ __| __/ _` |              *\n\
*          | |_| |  |  _ < (_) \__ \  __/ |_| || (_| |              *\n\
*           \__|_|  |_| \_\___/|___/\___|\__|\__\__,_|              *')
    print('*                                                                   *')
    print("* J Yang et al, Improved protein structure prediction using         *\n* predicted interresidue orientations, PNAS, 117: 1496-1503 (2020)  *")
    print("* Please email your comments to: yangjy@nankai.edu.cn               *")
    print('*********************************************************************')


def rosetta_main(bins_datas, args, params):
    ########################################################
    # process inputs
    ########################################################

    logo()
    # read params
    
    for num, bins_data in enumerate(bins_datas):
    
        

        # get command line arguments
        print(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        # init PyRosetta
        init('-mute all -hb_cen_soft  -relax:dualspace true -relax:default_repeats 3 -default_max_cycles 200 -detect_disulf -detect_disulf_tolerance 3.0')

        # Create temp folder to store all the restraints
        tmpdir = tempfile.TemporaryDirectory(prefix=params['WDIR'] + '/')
        params['TDIR'] = tmpdir.name
        print('temp folder:     ', tmpdir.name)

        # read and process restraints & sequence
        npz = bins_data
        seq = read_fasta(args.fasta)
        nres = len(seq)
        params['seq'] = seq
        rst = gen_rst(npz, tmpdir, params)

        best_e = float('inf')
        best_pose = None

        for re_iter in range(5):
        ########################################################
        # Scoring functions and movers
        ########################################################
            sf = ScoreFunction()
            sf.add_weights_from_file(scriptdir + '/rosetta_min/data/scorefxn.wts')

            sf1 = ScoreFunction()
            sf1.add_weights_from_file(scriptdir + '/rosetta_min/data/scorefxn1.wts')

            sf_vdw = ScoreFunction()
            sf_vdw.add_weights_from_file(scriptdir + '/rosetta_min/data/scorefxn_vdw.wts')

            sf_cart = ScoreFunction()
            sf_cart.add_weights_from_file(scriptdir + '/rosetta_min/data/scorefxn_cart.wts')

            mmap = MoveMap()
            mmap.set_bb(True)
            mmap.set_chi(False)
            mmap.set_jump(True)

            min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
            min_mover.max_iter(1000)

            min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.0001, True)
            min_mover1.max_iter(1000)

            min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.0001, True)
            min_mover_vdw.max_iter(500)

            min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
            min_mover_cart.max_iter(1000)
            min_mover_cart.cartesian(True)

            repeat_mover = RepeatMover(min_mover, 3)

            ########################################################
            # initialize pose
            ########################################################
            pose = pose_from_sequence(seq, 'centroid')

            # mutate GLY to ALA
            for i, a in enumerate(seq):
                if a == 'G':
                    # mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, 'ALA')
                    mutator = MutateResidue(i + 1, 'ALA')
                    mutator.apply(pose)
                    # print('mutation: G%dA'%(i+1))

            set_random_dihedral(pose)
            remove_clash(sf_vdw, min_mover_vdw, pose)

            ########################################################
            # minimization
            ########################################################
            print('\nenergy minimization ...')
            if args.mode == 0:

                # short
                print('short')
                add_rst(pose, rst, 1, 12, params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

                # medium
                print('medium')
                add_rst(pose, rst, 12, 24, params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

                # long
                print('long')
                add_rst(pose, rst, 24, len(seq), params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

            elif args.mode == 1:

                # short + medium
                print('short + medium')
                add_rst(pose, rst, 3, 24, params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

                # long
                print('long')
                add_rst(pose, rst, 24, len(seq), params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

            elif args.mode == 2:

                # short + medium + long
                print('short + medium + long')
                add_rst(pose, rst, 1, len(seq), params)
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)

            # mutate ALA back to GLY
            for i, a in enumerate(seq):
                if a == 'G':
                    # mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, 'GLY')
                    mutator = MutateResidue(i + 1, 'GLY')
                    mutator.apply(pose)
                    # print('mutation: A%dG'%(i+1))

            ########################################################
            # full-atom refinement
            ########################################################

            if str(args.fastrelax) == "True":

                scorefxn_fa = create_score_function('ref2015_cart')
                # scorefxn_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
                # scorefxn_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
                # scorefxn_fa.set_weight(rosetta.core.scoring.angle_constraint, 1)
                # scorefxn_fa.set_weight(rosetta.core.scoring.pro_close, 0.0)

                scorefxn_fa.set_weight(atom_pair_constraint, 5)
                scorefxn_fa.set_weight(dihedral_constraint, 1)
                scorefxn_fa.set_weight(angle_constraint, 1)
                scorefxn_fa.set_weight(pro_close, 0.0)


                mmap = MoveMap()
                mmap.set_bb(True)
                mmap.set_chi(True)
                mmap.set_jump(True)

                relax_round1 = rosetta.protocols.relax.FastRelax(scorefxn_fa, scriptdir + "/rosetta_min/data/relax_round1.txt")
                relax_round1.set_movemap(mmap)

                relax_round2 = rosetta.protocols.relax.FastRelax(scorefxn_fa, scriptdir + "/rosetta_min/data/relax_round2.txt")
                relax_round2.set_movemap(mmap)

                pose.remove_constraints()
                switch = SwitchResidueTypeSetMover("fa_standard")
                switch.apply(pose)

                print('\nrelax: First round ... (torsion space)')

                params['PCUT'] = 0.15
                add_rst(pose, rst, 1, nres, params, True)
                relax_round1.apply(pose)

                print('relax: Second round ... (cartesian space)')
                pose.remove_constraints()
                params['PCUT'] = 0.3
                add_rst(pose, rst, 1, nres, params, True)
                pose.conformation().detect_disulfides()  # detect disulfide bond again w/ stricter cutoffs
                relax_round2.apply(pose)

                # idealize problematic local regions
                idealize = rosetta.protocols.idealize.IdealizeMover()
                poslist = rosetta.utility.vector1_unsigned_long()

                scorefxn = create_score_function('empty')
                scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
                scorefxn.score(pose)

                emap = pose.energies()
                print("idealize...")
                for res in range(1, nres + 1):
                    cart = emap.residue_total_energy(res)
                    if cart > 50:
                        poslist.append(res)

                if len(poslist) > 0:
                    idealize.set_pos_list(poslist)
                try:
                    idealize.apply(pose)

                    # cart-minimize
                    scorefxn_min = create_score_function('ref2015_cart')
                    mmap.set_chi(False)

                    min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_min, 'lbfgs_armijo_nonmonotone', 0.00001, True)
                    min_mover.max_iter(100)
                    min_mover.cartesian(True)
                    # print("minimize...")
                    min_mover.apply(pose)

                except:
                    print('!!! idealization failed !!!')
                
                
            score_fn = create_score_function("ref2015").score
            
            e = score_fn(pose)

            if e < best_e:
                best_e = e
                best_pose = pose

        ########################################################
        # save final model
        ########################################################
        best_pose.dump_pdb(args.output + f'/{num}' + '.pdb')


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


def to_bins_main(sampled_6d):
    # parser = argparse.ArgumentParser(usage = " Convert backbone geometories value to probability distribution ")
    # parser.add_argument('--sample_dir', type=str, help='inference samples with 6d')
    
    # parser.add_argument("--out_ProbD", type=str,required=True, help="saved npz dir")
    # parser.add_argument('--tags', type=str, default='test', help='tags like pdbid same to save dir')
    # parser.add_argument('--symm', type=bool, default=True, help='symm dist and omega values')
    # parser.add_argument('--sigma', type=float, default=1, help='gaussian distribution sigma, treated not single value but distribution')
    # args = parser.parse_args()


    # coords_path = args.sample_dir   
    # coords_path = Path(coords_path)
    # sampled_6d_paths = os.listdir(coords_path)
    # out_path = args.out_ProbD
    # #workdir = Path('sampling','toRosetta', args.tags)
    # workdir = Path(out_path,'toRosetta')
    # print(workdir)
    # workdir.mkdir(parents=True, exist_ok=True)

    npz_list = []
    
    for ith, coords_6d in enumerate(sampled_6d):
        

        # with open(coords_6d_path, 'rb') as f:

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

        sym = True ## TODO: make input feature sym.  should use cb_pwd = cb_pwd.fill_diagonal_(0) set diag 0
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
        
        sigma = 1
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

        # npz_file = workdir.joinpath((f'sample2rosetta_{ith}'))
        # if os.path.dirname(npz_file):
        #     os.makedirs(os.path.dirname(npz_file), exist_ok=True)

        # np.savez_compressed(npz_file, **save_data)     # 每个都是 [N, N, 37].
        npz_list.append(save_data)

    return npz_list
        



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


def main(args):
    
    
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    batch_size = args.batch_size
    workdir = Path(args.output,"Sampling")
    print(f'Saved 6D rst in {workdir}')
    workdir.mkdir(parents=True, exist_ok=True)

    # Initialize model.
    score_model = get_model(config)
    loaded_state = torch.load(args.checkpoint, map_location=config.device)
    score_model.load_state_dict(loaded_state)

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
    
    all_results = []

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
            
            sample, _ = sampling_fn(score_model, condition=condition, context=context, esm_contact=esm_contact, infer=True) # [N, 5, 128, 128]

            generated_samples.append(sample.cpu())
            generated_samples = torch.cat(generated_samples, 0)  # [N, 5, 128, 128]

            # save the results.
            for j in range(batch_size):
                all_results.append(generated_samples[j].unsqueeze(0).numpy())
    
    return all_results

if __name__ == "__main__":
    
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(scriptdir + '/rosetta_min/data/params.json') as jsonfile:
            params = json.load(jsonfile)
    
    parser = argparse.ArgumentParser(usage = "Protein conformational ensembles generating")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file')
    parser.add_argument('--fasta', type=str, required=True,help="input sequence")
    parser.add_argument('--output', type = str, help = 'The directory where the output results are stored(*.npy).')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_samples_eval', type=int, default=100)
    parser.add_argument('--task', type=str, default='trRosetta', help='lst text save folder',choices=['trRosetta', 'ProteinSGM',])
    
    parser.add_argument('-pd', type=float, dest='pcut', default=params['PCUT'], help='min probability of distance restraints')
    parser.add_argument('-m', type=int, dest='mode', default=2, choices=[0,1,2], help='0: sh+m+l, 1: (sh+m)+l, 2: (sh+m+l)')
    parser.add_argument('-w', type=str, dest='wdir', default=params['WDIR'], help='folder to store temp files')
    parser.add_argument('-n', type=int, dest='steps', default=1000, help='number of minimization steps')
    parser.add_argument('--orient', dest='use_orient', action='store_true', help='use orientations')
    parser.add_argument('--no-orient', dest='use_orient', action='store_false')
    parser.add_argument('--fastrelax', dest='fastrelax', action='store_true', help='perform FastRelax')
    parser.add_argument('--no-fastrelax', dest='fastrelax', action='store_false')
    parser.add_argument('--log', dest='log', default='/public/home/wangwk/db/denovo_ss_new/trx_timing')
    parser.add_argument('--gpu', dest='gpu', default=-1,type=int)
    
    parser.set_defaults(use_orient=True)
    parser.set_defaults(fastrelax=True)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    params['PCUT'] = args.pcut
    params['USE_ORIENT'] = args.use_orient
    
    data_list = main(args)
    
    if args.task == 'trRosetta':
        output_data = to_bins_main(data_list)
        rosetta_main(output_data, args, params)
    elif args.task == 'ProteinSGM':
        proteinsgm_main(data_list, args)
    else:
        print("ERROR of task, please check and retry...")