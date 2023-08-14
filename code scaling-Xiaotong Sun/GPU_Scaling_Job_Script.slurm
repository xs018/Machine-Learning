#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --export=ALL
#SBATCH --gres=gpu:4
#SBATCH -t 0:20:00

cd $PROJECT/Scaling/

echo "running code"
singularity exec --nv /ocean/containers/ngc/tensorflow/tensorflow_latest.sif python3 ResNet50_gpu_v3.py -n 4 -b 2048
echo "done!"

## Job script for bridges
