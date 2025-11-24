#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=test-import-%j.out

source ~/.bashrc
conda activate chiron
cd /cluster/work/boeva/shoenig/Chiron3D

python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
python -c "from transformers.modeling_layers import GradientCheckpointingLayer; print('GradientCheckpointingLayer OK')"
