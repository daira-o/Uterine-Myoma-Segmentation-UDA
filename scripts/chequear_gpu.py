import torch

print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA versión:", torch.version.cuda)

# Test real: crea un tensor en GPU
x = torch.ones(3, 3).cuda()
print("Tensor en:", x.device)  # debe decir cuda:0