import torch

tensor1 = torch.tensor([[1, 2],
                        [2, 2],
                        [3, 2]]) 

tensor2 = torch.ones(1000, 2)  



result = torch.sub(tensor1[:, 0].unsqueeze(1), tensor2[:, 0].unsqueeze(0))
print(result)