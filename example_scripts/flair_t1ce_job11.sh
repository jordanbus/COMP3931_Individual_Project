#$ -cwd -V
#$ -l h_rt=04:30:00
#$ -l h_vmem=15G
#$ -m be
#$ -hold_jid 4915544
module load anaconda
source activate sc20jwb_project
python ./multilabel.py 11 20 100 6 53 0.3 0 0 flair t1ce
conda deactivate