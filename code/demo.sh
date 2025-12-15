
# Install PyRosetta from unpacked package
# tar jxvf *.bz2
# cd /data/PyRosetta/PyRosetta4.Release.python310.linux.release-385/setup
# python setup.py install
# cd /code

# Test demo protein
python test.py --fasta /data/demo/example.fasta --output /results/demo --config /data/demo/sequence_cond.yml --checkpoint ../data/checkpoints/Mac-Diff-128.pt  --batch_size 1 --num_samples_eval 1 --task trRosetta

# Test Homeodomain protein in Task I
python test.py --fasta ../data/FFPs/PROTEIN_B.fasta --output /results/FFPs/PROTEIN_B  --config ./configs/sequence_cond_128.yml --checkpoint ../data/checkpoints/Mac-Diff-128.pt  --batch_size 10 --num_samples_eval 10 --task trRosetta
echo "Sampling finished for Homeodomain"

# # Test Adk protein in Task II
python test.py --fasta ../data/ADK/ADK.fasta --output /results/ADK  --config ./configs/sequence_cond_256.yml --checkpoint ../data/checkpoints/Mac-Diff-256.pt  --batch_size 1 --num_samples_eval 1 --task ProteinSGM
echo "Sampling finished for Adk"


# # Cal TM-score between reference and generated structures
# chmod +x /code/scripts/TMscore
# # Closed state
# python /code/scripts/cal_tmscore.py /data/reference_data/Adk/1akeA.pdb /results/ADK/
# # Open state
# python /code/scripts/cal_tmscore.py /data/reference_data/Adk/4akeA.pdb /results/ADK/