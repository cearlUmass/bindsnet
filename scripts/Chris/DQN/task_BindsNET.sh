#!/bin/bash
#SBATCH --job-name=DiffuGA_BindsNET
#SBATCH --time=02-23:53:00
#SBATCH --mem=32768
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@30
#SBATCH --output=logs_task_BindsNET/slurm.%j.out
#SBATCH -p preempt

# trap at https://hpc-discourse.usc.edu/t/signalling-a-job-before-time-limit-is-reached/314/3

script_to_run="$1"
trap 'echo signal recieved!; kill "${PID}"; wait "${PID}"; handler' USR1 SIGINT SIGTERM

# Activate conda env
source /cluster/tufts/levinlab/hhazan01/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate chriscode
unset DISPLAY
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/tufts/levinlab/hhazan01/miniconda3/lib/
# module load ffmpeg/3.2.2

# load Cuda 11 
# module load cuda/12.2

echo "-------------1---------------"
echo $SLURMD_NODENAME
echo "-------------2---------------"
nvidia-smi
echo "-------------3---------------"
printenv CUDA_VISIBLE_DEVICES
echo "-------------4---------------"

echo "Job started!"

# load the command from file
line=''
while read line; do 
echo "read"
done < ${script_to_run}
echo $line

# run python script
python3 $line &

# delete command file
rm ${script_to_run}
PID="$!"
wait "${PID}"

echo "Job ended!"