#!/bin/bash
#SBATCH --job-name=ctd_pipeline
#SBATCH --partition=sc-freegpu
#SBATCH --account=soal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/sailhome/kadenwu/contingency_table_diffusion/outputs/slurm_%j.out
#SBATCH --error=/sailhome/kadenwu/contingency_table_diffusion/outputs/slurm_%j.err

# Full pipeline (make-dataset -> train-h -> sample-guided): use_pretraining=False,
# use_bk_regularization=True, bk_loss_weight=0.01, terminal_loss_weight=0.001
# (all set in config.py). make-dataset loads the EXISTING dataset at
# outputs/h_dataset.pt (same seed=0) rather than regenerating it, since
# cmd_train_h only regenerates when that file is missing. train-h always
# initializes a fresh HModel from scratch (train_h_model does this whenever
# no model= is passed in, which is what main.cmd_train_h does when
# use_pretraining=False) -- never resumes checkpoints/h_model_last.pt or
# h_model_best.pt.

cd /sailhome/kadenwu/contingency_table_diffusion
source venv/bin/activate
python main.py pipeline
