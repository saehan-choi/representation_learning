

class CFG:
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [16, 32, 64, 128, 256]


# for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels):
#     print(in_channel)
#     print(out_channel)

# # [in_channel, out_channel for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels)]


print(list(reversed(CFG.in_channels)))