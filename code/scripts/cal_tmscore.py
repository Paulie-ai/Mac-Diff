from pathlib import Path
import subprocess, os
import pandas as pd
from tqdm import tqdm
import argparse

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)

def tmscore(X_path,Y_path,):
    if not os.path.isabs(X_path): X_path = os.path.join(os.getcwd(), X_path) # ref
    if not os.path.isabs(Y_path): Y_path = os.path.join(os.getcwd(), Y_path)
    out = subprocess.check_output(['/code/scripts/TMscore', '-seq', Y_path, X_path], 
                    stderr=open('/dev/null', 'w'), cwd=my_dir)
    
    # print("out is ", out)
    start = out.find(b'RMSD')
    end = out.find(b'rotation')
    out = out[start:end]
    rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')    
    rmsd = float(rmsd.split(b'=')[-1])
    tm = float(tm.split(b'=')[1].split()[0])
    return {'rmsd': rmsd, 'tm': tm}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_path", type=str, help="which contain ref states")
    parser.add_argument("target_dir", type=str, help="dir which contain sampling protein conformations")
    parser.add_argument("--methods", type=str, default='Mac-Diff', help="Which methods used")
    args = parser.parse_args()
    ref_path = args.ref_path
    sampled_dir = args.target_dir
    out_dict = {}
    for i in tqdm(Path(sampled_dir).rglob('*.pdb')):
        outs = tmscore(Path(ref_path), i)
        out_dict['id_' + i.name.split('.')[0]] = outs

    df_to_refpdb = pd.DataFrame.from_dict(out_dict, orient='index')
    print(f"Sorted sampling results to {ref_path.split('/')[-1].split('.')[0].upper()} is:")

    sorted_rmscore = df_to_refpdb.sort_values(by='tm', ascending=False)
    print('By tmscore:', sorted_rmscore.head(), '\n')
