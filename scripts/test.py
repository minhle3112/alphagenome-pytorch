import torch
from alphagenome_pytorch import AlphaGenome

# Correct function for the Kundaje Lab PyTorch version
model = AlphaGenome.from_pretrained("/srv/scratch/wonglab/minh/weights/model_all_folds.safetensors")

# This will print the full architecture
print(model)