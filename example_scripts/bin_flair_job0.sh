#$ -cwd -V
#$ -l h_rt=04:00:00
#$ -l h_vmem=15G
#$ -m be
module load anaconda
source activate sc20jwb_project
python ./binary_classifier.py 0 20 100 6 42 0.3 flair
conda deactivate
