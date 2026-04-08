import torch
from alphagenome_pytorch import AlphaGenome

# Correct function for the Kundaje Lab PyTorch version
model = AlphaGenome.from_pretrained("/srv/scratch/wonglab/minh/weights/model_all_folds.safetensors")

# This will print the full architecture
print(model)

# 1. Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# 2. Calculate TRAINABLE parameters (Crucial for LoRA)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params:,}")

# 3. Calculate the percentage of trainable parameters
percentage = 100 * trainable_params / total_params
print(f"Percentage Trainable: {percentage:.4f}%")