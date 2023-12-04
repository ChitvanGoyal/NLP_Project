#!/bin/bash
#
#SBATCH --job-name=bert_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=10:30:00
#SBATCH --output=bert_finetune.out
#SBATCH --error=bert_finetune.err

module purge

singularity exec --nv --overlay /scratch/cg4174/nlp_project/env/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash << EOF

source /ext3/env.sh
python bert_finetune.py
EOF
