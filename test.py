import torch

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
all_dataset = np.array([[1,2,3,4,6,7,10,8,8,4],
                [1,2,3,2,6,7,8,2,8,4],
                [1,2,2,3,6,7,2,8,6,2],
                [1,2,3,3,6,4,4,2,2,6],
                [1,2,5,4,6,7,8,2,8,1],
                [1,2,3,3,6,1,8,4,8,5]])


image = np.array([5,12,3,2,4,6,2,4,5,6])

# a3 = (a1-a2)**2
# print(np.sqrt(a3.sum(axis=1)))


# 정규화 + 후에 np.sqrt(k^2)
# 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.


# def k_nearest_neighbor(image, all_dataset, labels, K):

#     knn = NearestNeighbors(n_neighbors=2, metric='cosine')
#     knn.fit(image)
#     # 필요시 axis 바꾸세요 ㅎㅎ

#     _, indices = knn.kneighbors(all_dataset)
#     indices_list = indices.tolist()

#     print(f'indices_list:{indices_list}')









# def k_nearest_neighbor(image, all_dataset, labels, K):
#     # 정규화 + 후에 np.sqrt(k^2)
#     # 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.
#     square = (all_dataset-image)**2

#     distance = np.sqrt(square.sum(axis=1))
#     # print(distance)
#     # print(distance.shape)

#     sorted_idx = np.argsort(distance)[:K]
#     # 제일작은 index를 반환함

#     for i in sorted_idx:
#         similarImgPath = labels[i]
#         print(similarImgPath)

# labels = ['./dataset/mask1.jpg', './dataset/123.jpg', './dataset/634125.jpg', './dataset/162.jpg', './dataset/6131.jpg', './dataset/63127.jpg']

# # 이런식으로 우선 개발진행해보자
# k_nearest_neighbor(image, all_dataset, labels, K=5)


# embedding을 이해하려면 이거 돌려보면됨.  reshape안해도 괜찮긌네
# k = torch.randn(2,1,3,3)

# print(f'k:{k}')
# print(f'k:{k.size()}')


# m = k.flatten(start_dim=1)
# print(m)
# print(m.size())

