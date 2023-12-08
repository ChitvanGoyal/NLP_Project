#!/bin/bash
#
#SBATCH --job-name=query2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=14:30:00
#SBATCH --output=query2.out
#SBATCH --error=query2.err

module purge

singularity exec --nv --overlay /scratch/cg4174/nlp_project/env/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash << EOF

source /ext3/env.sh
python query2.py
EOF
