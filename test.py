import torch

class CFG:
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [16, 32, 64, 128, 256]


# for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels):
#     print(in_channel)
#     print(out_channel)

# # [in_channel, out_channel for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels)]


# print(list(reversed(CFG.in_channels)))



# print([1,2,3,4,5]+[1,1,1,1,1,5,3,6,1,2])
# 뒤로 배열됨.



print(torch.randn(100).size())