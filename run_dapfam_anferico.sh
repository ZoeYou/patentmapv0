#!/bin/bash
#SBATCH --job-name=dapfam_anferico
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=you.zuo@inria.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=almanach
#SBATCH --nodelist=gpu01[2-3],gpu01[5-7]
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=res_dapfam_anferico_%j.out
#SBATCH --error=res_dapfam_anferico_%j.out
#SBATCH -A almanach

echo "### Running ${SLURM_JOB_NAME} on $(hostname) ###"
cd ${SLURM_SUBMIT_DIR}

# Activate conda env
source /home/$USER/.bashrc
conda activate patentmap

# Stage-W3 sanity check: run DAPFAM only on anferico/bert-for-patents
# (skips perf200 retrieval to save time; IPC classification still runs as it is
# unconditional in evaluate.py).
python evaluate.py \
    --model_name anferico/bert-for-patents \
    --benchmark dapfam
