import torch
import torch.nn

from torch.utils.data import DataLoader

import tqdm
from representation_learning import *

import numpy as np
import os

# 여기서 변경잠시 하겠습니다.
CFG.trainPath = glob('./dataset2/train/*.jpg')

CFG.batch_size = 128
one_image = './dataset2/train/21.jpg'
num_sample = 10
# 몇개의 비슷한 이미지들을 뽑아낼 것인지

class ImgDataset(Dataset):
    def __init__(self, image_list, transform):
        # 여기서는 딱히 나눌필요 없어보임 왜냐면, representation learning 을 이용하기 때문
        # label도 필요없네요.
        super(ImgDataset, self).__init__()
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        img_name = image.split('\\')[-1]

        image = cv2.imread(image, cv2.IMREAD_COLOR)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            pass

        # label을 지정하면 안될듯 ?
        return image, img_name

def nearest_neighbor(embedding_image, all_embedding_images, labels, num_sample):
    # num_sample개의 이미지 path만 추출

    # 정규화 + 후에 np.sqrt(k^2)
    # 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.
    square = (all_embedding_images-embedding_image)**2

    distance = np.sqrt(square.sum(axis=1))
    
    sorted_idx = np.argsort(distance)[:num_sample]
    # 제일작은 index를 반환함

    for i in sorted_idx:
        similarImgPath = labels[i]
        print(similarImgPath)
        # 여기서 비슷한 이미지들의 이름이 주어집니다.




def create_embedding(encoder, dataloader, embedding_dim, device):
    """
    create embedding using encoder from dataloader.
    encoder: A convolutional Encoder. e.g. torch_model convEncoder
    dataloader: PyTorch datalodaer, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimensions.
    returns: embedding of size (num_image_in_loader+1, c, h, w) --> ㅇㅎ!
    """
    # 처음 embedding에 쓰레기값이 들어가므로 여기서도 쓰레기값 집어넣었습니다.
    labels = ['NOTHING']
    embedding = torch.randn(embedding_dim)

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    with torch.no_grad():
        for step, data in bar:
            images = data[0].to(device, dtype=torch.float)
            
            for label in data[1][:]:
                labels.append(label)
            
            # 라벨들을 배치사이즈로 받기위해서 이렇게 시도합니다.
            encode_outputs = encoder(images).cpu()
            # 여기 쓰레기값이 들어가므로 본래 이미지보다 한개 더 많다.
            embedding = torch.cat((embedding, encode_outputs), 0)

    # 굳 여기에 라벨 데리고 왔습니다. ㅎ
    return embedding, labels



encodeModel = encoderModel().to(CFG.device)
encodeModel.load_state_dict(torch.load('./weights/encoder_model.pt'))
encodeModel.eval()

trainDataset = ImgDataset(CFG.trainPath, CFG.transform)
trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=CFG.batch_size)

embedding_shape = (1, 1024, int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))), int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))))
# 이거 마지막으로 나오는 encoder dimension 인데 쓰레기값으로 들어감
# 마지막 네트워크의 (1, c, w, h) 입니다. batch가 1인이유는 concatenate를 위한 trash value입니다.

embedding, labels = create_embedding(encodeModel, trainDataloader, embedding_shape, CFG.device)
all_embedding_images = torch.flatten(embedding, start_dim=1).cpu().detach().numpy()
# 아! 여기서 제일 L2 distance(euclidean distance)가 제일 낮은애들 몇개가 나올거잖아요
# -> 걔내들만 추려서 원래의 embedding vector들과 비교해서 알아내도록 하는게 제일 나은선택 같습니다.
# 걔내들의 이미지를 알아내면 그 경로의 이미지들을 shutil.copy 하면 될 듯 합니다.




# print(all_embedding_images)
image = cv2.imread(one_image, cv2.IMREAD_COLOR)
transformed = CFG.transform(image=image)
image = transformed['image'].unsqueeze(0).to(CFG.device, dtype=torch.float)

one_embedding_image = encodeModel(image).flatten(start_dim=1)
one_embedding_image = one_embedding_image.detach().cpu().numpy()

print(one_embedding_image.shape)

nearest_neighbor(one_embedding_image, all_embedding_images, labels, num_sample)
# 비슷한이미지를 찾아내기 위해 이미지를 넣어줌.
# 여기서 제일 비슷한 애들을 찾아낼 것임.