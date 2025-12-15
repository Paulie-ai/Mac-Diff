# Mac-Diff: Modal-aligned conditional Diffusion
## Introduction 

<p>Conditional Diffusion with Locality Aware Modal Alignment for Generating Diverse Protein Conformational Ensembles.</p>


## Installation

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
<pre>
<code>Command:
> cd Mac_Diff/code # Enter the current directory
> conda create -n mac_diff python=3.11
> conda activate mac_diff
> conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
> pip install einops==0.8.0 scipy==1.14.1 deepspeed==0.14.0 dm-tree==0.1.8 ml-collections==1.1.0 diffusers==0.34.0 easydict==1.13 fair-esm==2.0.0
> wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
> pip install flash_attn-2.6.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
</code>
</pre>
#### * Note : The installation of pytorch and flash-attention needs to correspond to your system, GPU and other versions, please check https://pytorch.org/get-started/locally/ and https://github.com/Dao-AILab/flash-attention. 

### PyRosetta installation

This project requires [PyRosetta](https://www.pyrosetta.org). Please follow the steps below:
  
```bash
# Release precompiled package installation (recommended). 
# Download the PyRosetta package from https://www.pyrosetta.org/downloads and install as follows: 
> wget https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux/PyRosetta4.Release.python310.linux.release-385.tar.bz2 
> tar jxvf PyRosetta4.Release.python310.linux.release-385.tar.bz2
> cd PyRosetta4.Release.python310.linux.release-385/setup
> python setup.py install

# (Optional) Use a conda environment:
> conda config --add channels https://USERNAME:PASSWORD@conda.graylab.jhu.edu #Requires registration for password
> conda install pyrosetta
# Test installation:
> python -c "from pyrosetta import init; init()"
```

We provide two weight files of Mac-Diff.

* **Mac-Diff-128** — pretrained on PDB and then fine-tuned on MD trajectories.  
* **Mac-Diff-256** — same training setting as Mac-Diff-128, with training limited to sequences of length 256.

#### Weights URL: https://drive.google.com/drive/folders/10lwdDgqTSA_GOGwNFpHrzA_TQETSIRQY?usp=drive_link
<!-- 
### Mac-Diff models
| Model|Weights URL |
|:---|:--|
| Mac-Diff  |  https://drive.google.com/drive/folders/10lwdDgqTSA_GOGwNFpHrzA_TQETSIRQY?usp=drive_link | -->


## Usage

### 1. Demo of making inference.
<p>The well-trained models are saved in <code>/data/checkpoints/*</code>. These models were trained under two types of training datasets (MD trajectories Data and Protein Structure Data in the PDB). <code>test.py</code> is used for making predictions using well-trained models. </p>

<pre>
<code>Usage: python test.py [options]
Required:
--fasta STRING: choose input protein sequence files(*.fasta).
--output STRING: the directory to save the prediction results, output as npy files.
--config STRING: the path to the configuration settings file (*.yml). 
--checkpoint STRING: the path to the model weights files(*.pt).

Optional:
--batch_size STRING: the numbers of samples processed during prediction (default:1). # Note that the batch size can be chosen based on your GPU memory and count.
--num_samples_eval STRING: total numbers of samples used during evaluation (default:1).
</code> </pre>

We provide a small demo example to test the installation of the Mac-Diff and to show you how to use Mac-Diff to inference the protein backbone geometories.

<pre>
<code>python test.py --fasta /data/demo/example.fasta --output /results/demo --config /data/demo/sequence_cond.yml --checkpoint ../data/checkpoints/Mac-Diff-128.pt  --batch_size 1 --num_samples_eval 1 --task trRosetta
</code></pre>

* --fasta *.fasta file contains of a description line starting with > followed by the protein sequence(the 20 standard amino acids) itself. 
* --output *.npy file contains a 128 × 128 × 5 tensor, with four channels correspond to the values of distance, omega, theta, phi angles and one padding channel indicating sequence length.
* --config *.yml file contains parameters for 'training', 'sampling', 'data', 'model', and 'optim', etc.

### 2. Evaluate Mac-Diff on test datasets with well-trained models.

<p> 1) <code>test.py</code> is used for sampling protein backbone geometries for fast folding proteins and BPTI. 

Use protein Homeodomain as an example: </p>
<pre>
<code># Fast folding proteins
python test.py --fasta ../data/FFPs/PROTEIN_B.fasta --output /results/FFPs/PROTEIN_B  --config ./configs/sequence_cond_128.yml --checkpoint ../data/checkpoints/Mac-Diff-128.pt  --batch_size 10 --num_samples_eval 10 --task trRosetta
</code></pre>
* The benchmark MD trajectories of 12 fast folding proteins (<a href="https://www.science.org/doi/10.1126/science.1208351">Shaw et al.</a>) and BPTI (<a href="https://www.science.org/doi/10.1126/science.1187409">Lindorff-Larsen et al.</a>) can be obtained from the original authors in D.E. Shaw Research (https://www.deshawresearch.com).

<!-- <p> 2) Then, recover the protein structures from each protein backbone geometries *.npy file by using <code>convert2trRosetta_bins.py</code> and <code>script2rosetta.py</code>: </p>

<pre>
<code># Convert denoised pairwise geometric representation to probability distribution of every residue pair by a Gaussian function.
python convert2trRosetta_bins.py --sample_dir ../results/FFPs/Homeodomain/Sampling/ --out_ProbD ../results/FFPs/Homeodomain/toRosetta

# Prepare the required files to trRosetta
python script2rosetta.py --rosetta_input ../results/FFPs/Homeodomain/toRosetta --fasta ../data/FFPs/HOMEODOMAIN.fasta --pred_folder ../results/FFPs/Homeodomain/Rosetta_outputs -->
<!-- </code></pre>


<p> 3) Using <code>trRosetta</code> scripts (<a href="https://www.pnas.org/doi/abs/10.1073/pnas.1914677117">developed by Yang et al.</a>)  with the <code>pyrosetta</code> package to recover the all-atom protein structure: </p> 

<pre>
<code>> cd ../trRosetta/
> cat run.txt | parallel -j 10 --no-notice -->
</code></pre>


### 3. Evaluate Mac-Diff on ADK and Cfold40 test set.
<p> Using Adk (reference PDB ID: 1AKE/4AKE)as an example and reference scripts in ProteinSGM (<a href="https://www.nature.com/articles/s43588-023-00440-3">developed by Lee et al.</a>) with the <code>pyrosetta</code> package to recover the all-atom protein alternative structure: </p> 
<pre>
<code>
# Sampling protein geometries
> python test.py --fasta ../data/ADK/ADK.fasta --output /results/ADK  --config ./configs/sequence_cond_256.yml --checkpoint ../data/checkpoints/Mac-Diff-256.pt  --batch_size 1 --num_samples_eval 1 --task ProteinSGM
<!-- # Prepared the required files
> python script2rosetta.py --rosetta_input ../data/ADK/Sampling --fasta ../data/ADK/ADK.fasta --pred_folder ../results/ADK/Rosetta_outputs --out ProteinSGM
> cd ../ProteinSGM/
> cat run.txt | parallel -j 10 --no-notice -->
</code></pre>

#### * Note that the reference protein structures are available at the PDB. 

### 4. Analysis
<p> TM-score calculated between the reference and generated conformations. </p>
<pre><code> 
chmod +x /code/scripts/TMscore
python /code/scripts/cal_tmscore.py /data/reference_data/Adk/1akeA.pdb /results/ADK/
python /code/scripts/cal_tmscore.py /data/reference_data/Adk/4akeA.pdb /results/ADK/
</code></pre>

##### * Note: The generated PDB files are saved to the /results directory in Code Ocean, which is write-only during execution. So we recommend running TM-score locally to evaluate the results.

For further analysis of the results, please refer to the evaluation scripts available at https://github.com/lujiarui/Str2Str and https://github.com/bjing2016/EigenFold.
