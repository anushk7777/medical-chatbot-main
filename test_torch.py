import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Custom Ops:", hasattr(torch, '_custom_ops'))