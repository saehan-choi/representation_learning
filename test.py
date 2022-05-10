import torch

import numpy as np


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



# print(torch.randn(100).size())


# arr = [1,2,3,4,5,6]

# print(max(arr[:-1]))

# a = torch.randn(1,3,512,512)
# b = torch.randn(1,3,512,512)


# print(a.size())

# k = torch.cat((a,b), 0)
# print(k.size())
a1 = np.array([[1,2,3,4,6,7,10,8,8,4],
                [1,2,3,2,6,7,8,2,8,4],
                [1,2,2,3,6,7,2,8,6,2],
                [1,2,3,3,6,4,4,2,2,6],
                [1,2,5,4,6,7,8,2,8,1],
                [1,2,3,3,6,1,8,4,8,5]])


a2 = np.array([5,12,3,2,4,6,2,4,5,6])

a3 = (a1-a2)**2
print(np.sqrt(a3.sum(axis=1)))



def k_nearest_neibor(input, dataset, labels, K):
    # k = a-b
    # a = embedding_image  b = search_image
    # root((a[0]-b[0])^2+(a[1]-b[1])^2+(a[2]-b[2])^2)
    # 이런방식으로 진행해야 합니다.

    # 정규화 + 후에 np.sqrt(k^2)
    # 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.

    square = (dataset-input)**2
    distance = np.sqrt(square.sum(axis=1))
    print(f'dists:{distance}')

k_nearest_neibor()