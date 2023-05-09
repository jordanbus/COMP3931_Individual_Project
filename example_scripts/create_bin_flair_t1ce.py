import sys

# Get command line arguments
if len(sys.argv) < 3:
    print("Usage: python create_script.py job_index hold_jid")
    sys.exit(1)

job_index = sys.argv[1]
hold_jid = sys.argv[2]
seed = 42 + int(job_index)

# Template for the file content
template = '''#$ -cwd -V
#$ -l h_rt=04:00:00
#$ -l h_vmem=15G
#$ -m be
#$ -hold_jid [hold_jid]
module load anaconda
source activate sc20jwb_project
python ./binary_classifier.py [job_index] 20 100 6 [seed] 0.3 flair t1ce
conda deactivate'''

# Replace placeholders with command line arguments
content = template.replace("[job_index]", job_index).replace("[hold_jid]", hold_jid).replace("[seed]", str(seed))

# Write content to file
filename = "bin_flair_t1ce_job{}.sh".format(job_index)
with open(filename, "w") as f:
    f.write(content)

print("Script file created as '{}'".format(filename))
