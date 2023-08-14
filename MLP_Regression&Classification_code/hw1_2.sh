#!/bin/bash
######## --send email ########
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=xs018@uark.edu

######## Job Name: Train_Job ########
#SBATCH -J HW1_Job
#SBATCH -o log/HW1_Job.o%j
#SBATCH -e log/HW1_Job.e%j

#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00

module load AI/anaconda3-tf2.2020.11
conda activate /jet/home/xs018/envs
cd /jet/home/xs018/code

python hw1_2.py