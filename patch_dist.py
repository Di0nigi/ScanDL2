# patch_dist.py
import os
import platform

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# Force gloo backend on Windows
if platform.system() == 'Windows':
    os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

print("reached")