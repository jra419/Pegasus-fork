import torch

# model = torch.load_state_dict(torch.load('H.pth'))
model = torch.load('H.pth')

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print(model)
