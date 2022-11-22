#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="P100|V100"
#SBATCH --account=ie-idi
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="self_cond_wikitext2"
#SBATCH --output=training_output.out
#SBATCH --mail-user=williad@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

LOG_NAME="wikitext2_self_cond"

PROJECT_ROOT="/cluster/work/williad/text-diffusion"


mv training_output.out ${PROJECT_ROOT}/training_output.out # Move to job dir
cd ${PROJECT_ROOT}

# Remove log if name already exists

# if [[ -f logs/training/${LOG_NAME} ]]; then 
echo "Deleting log folder with same name ${LOG_NAME}"
rm -rf logs/${LOG_NAME}
# fi



echo "we are running from this directory: $PROJECT_ROOT"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Starting script"
echo ""
module purge
module load Anaconda3/2020.07
pip install torchtext
pip install torchdata

# Create log folder
echo "Creating log folder ${LOG_NAME} with path logs/${LOG_NAME}"
mkdir -p logs/${LOG_NAME}

# Move slurm output to log folder
mv ${PROJECT_ROOT}/training_output.out ${PROJECT_ROOT}/logs/${LOG_NAME}/training_output.out

python src/main.py --log_name ${LOG_NAME} --epochs 40000 --dataset wikitext2 --line_slice 2000 --num_layers 6 --nhead 5 --d_model 100 --embed_dim 100 --dim_feedforward 1024 --use_old_arch True --pretrained_emb glove
uname -a
