'''
give one dir and generat python lst
'''
from pathlib import Path
import argparse


def main():

    parser = argparse.ArgumentParser(usage = "Using TrRosetta protocol")
    parser.add_argument('--rosetta_input', type=str, help='all npz folder')
    parser.add_argument('--fasta', type=str, help='fasta file')
    parser.add_argument('--pred_folder', type=str, help='TrRosetta outputs folder')
    parser.add_argument('--out', type=str, default='trRosetta', help='lst text save folder',choices=['trRosetta', 'ProteinSGM',])
    parser.add_argument('--rank', type=int, default=0, help='devices')
    
    args = parser.parse_args()

    if args.out == "trRosetta":
        with open(f'../{args.out}/run.txt', 'w') as file:
            num = 0
            for i in Path(args.rosetta_input).rglob('*.npz'):

                script_str = f'CUDA_VISIBLE_DEVICES={args.rank} python trRosetta.py -NPZ {i} -FASTA {args.fasta} -OUT {args.pred_folder}/{num}.pdb'
                num +=1
                file.write(script_str + '\n')

        outdir = Path(args.pred_folder)
        outdir.mkdir(parents=True, exist_ok=True)

    elif args.out == "ProteinSGM":
        with open(f'../{args.out}/run.txt', 'w') as file:
            num = 0
            for i in Path(args.rosetta_input).rglob('*.npy'):

                script_str = f'python sampling_rosetta_v2.py {i} --fasta {args.fasta} --out {args.pred_folder}/{num}.pdb --tag {num}'
                num +=1
                file.write(script_str + '\n')

        outdir = Path(args.pred_folder)
        outdir.mkdir(parents=True, exist_ok=True)

    
if __name__ == "__main__":
    main()
