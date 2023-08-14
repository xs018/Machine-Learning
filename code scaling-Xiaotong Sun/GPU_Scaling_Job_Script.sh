#!/bin/bash
######## --send email ########
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=xs018@uark.edu
######## Job Name: Train_Job ########
#SBATCH -J GPU_Job
#SBATCH -o log/GPU8_512_Job.o%j
#SBATCH -e log/GPU8_512_Job.e%j

#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --export=ALL
#SBATCH --gres=gpu:8
#SBATCH -t 01:00:00

module load AI/anaconda3-tf2.2020.11
conda activate /jet/home/xs018/envs
cd /jet/home/xs018/code

python3 hw1_2.py -n 8 -b 512

## Job script for bridges
