import torch
import torch.nn

from torch.utils.data import DataLoader

import tqdm
from representation_learning import *

import numpy as np


def nearest_neighbor(image, all_dataset, labels, num_sample):
    # num_sample개의 이미지 path만 추출

    # 정규화 + 후에 np.sqrt(k^2)
    # 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.
    square = (all_dataset-image)**2

    distance = np.sqrt(square.sum(axis=1))
    # print(distance)
    # print(distance.shape)

    sorted_idx = np.argsort(distance)[:num_sample]
    # 제일작은 index를 반환함

    for i in sorted_idx:
        similarImgPath = labels[i]
        print(similarImgPath)


def create_embedding(encoder, dataloader, embedding_dim, device):
    """
    create embedding using encoder from dataloader.
    encoder: A convolutional Encoder. e.g. torch_model convEncoder
    dataloader: PyTorch datalodaer, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimentions.
    returns: embedding of size (num_image_in_loader+1, c, h, w) --> ?
    """
    encoder.eval()
    embedding = torch.randn(embedding_dim)

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    with torch.no_grad():
        for step, data in bar:
            images = data.to(device, dtype=torch.float)
            encode_outputs = encoder(images).cpu()
            # 여기 쓰레기값이 들어가므로 본래 이미지보다 한개 더 많다.
            embedding = torch.cat((embedding, encode_outputs), 0)
            # 휴,....... 이미지 이름을 어케가져오냐고...

    return embedding

encodeModel = encoderModel().to(CFG.device)
# decodeModel = decoderModel().to(CFG.device)
encodeModel.load_state_dict(torch.load(CFG.weightsavePath+'encoder_model.pt'))

trainDataset = ImgDataset(CFG.trainPath, CFG.transform)
valDataset = ImgDataset(CFG.valPath, CFG.transform)

trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=CFG.batch_size)
valDataloader = DataLoader(valDataset, shuffle=False, batch_size=CFG.batch_size)

embedding_shape = (1, 256, int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))), int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))))

# 이거 마지막으로 나오는 encoder dimension 인데 쓰레기값으로 들어감
embedding = create_embedding(encodeModel, trainDataloader, embedding_shape, CFG.device)
flattend_embedding = torch.flatten(embedding, start_dim=1).cpu().detach().numpy()

# 굳이 이거 저장하는거 필요없겠다.
# np.save(CFG.embeddingsavePath+"data_embedding.npy", flattend_embedding)
# embeddingPath = './embedding/data_embedding.npy'
# data = np.load(embeddingPath)
# print(data.shape)

print(flattend_embedding.shape)

# 무슨 이미지인지 알아야 나중에 plot 띄우지..
# 흠....................... 어케 해결할 것인가?
# embedding만 가지고 이미지를 알아낼 수 는 없을까?
# model을 돌려서 embedding을 저장한다음에 맞는지 비교하면 금방알아낼 수 있다.
# 다만 embedding을 만든 모델이 똑같아야 하고 가중치도 똑같아야합니다.

