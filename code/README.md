# Mac-Diff: Modal-aligned conditional Diffusion
## Introduction 

<p>Conditional Diffusion with Locality Aware Modal Alignment for Generating Diverse Protein Conformational Ensembles.</p>


## Installation

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
<pre>
<code>Command:
> cd Mac_Diff/code # Enter the current directory
> conda create -n mac_diff python=3.8
> conda activate mac_diff
> conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
> conda install -c conda-forge parallel
> pip install numpy==1.24.3 scipy==1.10.1 tqdm==4.65.0 yaml==0.2.5 pyyaml easydict biotite fair-esm
> wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.5/flash_attn-2.3.5+cu117torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
> pip install flash_attn-2.3.5+cu117torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
</code>
</pre>
#### * Note : The installation of pytorch and flash-attention needs to correspond to your system, GPU and other versions, please check https://pytorch.org/get-started/locally/ and https://github.com/Dao-AILab/flash-attention. 


## Model weights

We provide three weight files of Mac-Diff.

* **Mac-Diff-MD**&mdash;trained on all-atom, explicit solvent MD trajectories (ATLAS and GPCRMD).
* **Mac-Diff-PDB128**&mdash;trained on PDB structure to model protein conformational heterogeneity.
* **Mac-Diff-PDB256**&mdash;trained on PDB structures, with protein chains truncated to 256 residues to standardize input lengths.

### Mac-Diff models
| Model|Weights URL |
|:---|:--|
| Mac-Diff-MD  |  https://drive.google.com/file/d/12mP_W6r9YnRrb0Z2BOsH81CcuIjcKGm3/view?usp=sharing |
| Mac-Diff-PDB128 | https://drive.google.com/file/d/19ZjoJNYTHinBshTIyRtV8uIV0PUE65h0/view?usp=sharing  |
| Mac-Diff-PDB256 |  https://drive.google.com/file/d/1s3X7RvIJ53Iw1cIk94KdmjxK2jIQwYJH/view?usp=sharing |


## Usage

### 1. Demo of making inference.
<p>The well-trained models are saved in <code>/code/checkpoints/*</code>. These models were trained under two types of training datasets (1,674 MD trajectories and Protein Structure Data on or before May, 1st, 2022). <code>inference.py</code> is used for making predictions using well-trained models. </p>

<pre>
<code>Usage: python inference.py [options]
Required:
--fasta STRING: choose input protein sequence files(*.fasta).
--output STRING: the directory to save the prediction results, output as npy files.
--config STRING: the path to the configuration settings file (*.yml). 
--checkpoint STRING: the path to the model weights files(*.pt).

Optional:
--batch_size STRING: the numbers of samples processed during prediction (default:10). # Note that the batch size can be chosen based on your GPU memory and count.
--num_samples_eval STRING: total numbers of samples used during evaluation (default:10).
</code> </pre>

We provide a small demo example to test the installation of the Mac-Diff and to show you how to use Mac-Diff to inference the protein backbone geometories.

<pre>
<code>python inference.py --fasta ../data/demo/example.fasta --output ../data/demo/  --config ../data/demo/sequence_cond.yml --checkpoint ./checkpoints/mac_diff_ep3_best_20240929.pt  --batch_size 1 --num_samples_eval 1
</code></pre>

* --fasta *.fasta file contains of a description line starting with > followed by the protein sequence(the 20 standard amino acids) itself. 
* --output *.npy file contains a 128 × 128 × 5 tensor, with four channels correspond to the  values of distance, omega, theta, phi angles and one padding channel indicating sequence length.
* --config The *.yml file contains parameters for 'training', 'sampling', 'data', 'model', and 'optim', etc.

### 2. Evaluate Mac-Diff on test datasets with well-trained models.

<p> 1) <code>test.py</code> is used for sampling protein backbone geometries for fast folding proteins. 

Use protein Homeodomain as an example: </p>
<pre>
<code># Fast folding proteins
python test.py --fasta ../data/FFPs/HOMEODOMAIN.fasta --output ../data/FFPs/  --config ./configs/sequence_cond.yml --checkpoint ./checkpoints/mac_diff_ep3_best_20240929.pt  --batch_size 200 --num_samples_eval 1000
</code></pre>
* The benchmark MD trajectories of 12 fast folding proteins (<a href="https://www.science.org/doi/10.1126/science.1208351">Shaw et al.</a>) and BPTI (<a href="https://www.science.org/doi/10.1126/science.1187409">Lindorff-Larsen et al.</a>) can be obtained from the original authors in D.E. Shaw Research (https://www.deshawresearch.com).

<p> 2) Then, recover the protein structures from each protein backbone geometries *.npy file by using <code>convert2trRosetta_bins.py</code> and <code>script2rosetta.py</code>: </p>

<pre>
<code># Convert denoised pairwise geometric representation to probability distribution of every residue pair by a Gaussian function.
python convert2trRosetta_bins.py --sample_dir ../data/FFPs/Sampling/ --out_ProbD ../data/FFPs/

# Prepare the required files to trRosetta
python script2rosetta.py --rosetta_input ../data/FFPs/toRosetta/ --fasta ../data/FFPs/HOMEODOMAIN.fasta --pred_folder ../data/FFPs/Rosetta_outputs
</code></pre>


<p> 3) Using <code>trRosetta</code> scripts (<a href="https://www.pnas.org/doi/abs/10.1073/pnas.1914677117">developed by Yang et al.</a>)  with the <code>pyrosetta</code> package to recover the all-atom protein structure: </p> 

<pre>
<code>> cd ../trRosetta/
> cat run.txt | parallel -j 10 --no-notice # install pyrosetta first

</code></pre>


### 3. Evaluate Mac-Diff on ADK and Cfold40 test set.
<p> Using protein calbindin D9k (PDB ID: 4ICB_1IGV)as an example and reference scripts in ProteinSGM (<a href="https://www.nature.com/articles/s43588-023-00440-3">developed by Lee et al.</a>) with the <code>pyrosetta</code> package to recover the all-atom protein alternative structure: </p> 

<pre>
<code>
# Sampling protein geometries
> python test.py --fasta ../data/Cfold40/Seq_128/4ICB_1IGV.fasta  --output ../data/Cfold40/Seq_128/ --config ./configs/sequence_cond_PDB128.yml  --checkpoint ./checkpoints/PDB_128_20241016.pth --batch_size 1 --num_samples_eval 1

# Prepared the required files
> python script2rosetta.py --rosetta_input ../data/Cfold40/Seq_128/Sampling --fasta ../data/Cfold40/Seq_128/4ICB_1IGV.fasta --pred_folder ../data/Cfold40/Seq_128/Rosetta_outputs --out ProteinSGM
> cd ../ProteinSGM/
> cat run.txt | parallel -j 10 --no-notice

</code></pre>
#### * Note that the reference protein structures are available at the PDB. 

### 4. Analysis
For further analysis of the results, please refer to the evaluation scripts available at https://github.com/lujiarui/Str2Str and https://github.com/bjing2016/EigenFold.
