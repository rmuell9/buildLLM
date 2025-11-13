import torch

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

torch2d = torch.tensor([[1, 2], [3, 4]])

torch3d = torch.tensor([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])

print(tensor1d.dtype)  # int64

floatvec = torch.tensor([1.0, 2.0, 3.0])  # float32
print(floatvec.dtype)
floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)
print(floatvec)
