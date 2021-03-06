import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from glob import glob
import cv2
import numpy as np

import timm

# 간단한 네트워크부터 끝내고, 그 다음시작 ㄱㄱ

class CFG:
    # imagePath
    # 이거 jpg말고도 될수있게 한번ㄱ..ㄱ;
    weightsavePath = './weights/'
    embeddingsavePath = './embedding/'

    trainPath = glob('./dataset2/train/*.jpg')
    # valPath = glob('./dataset2/val/*.jpg')

    in_channels = [3, 64, 128, 256, 512]
    # 이거 네트워크 더 깊게하니깐 maxpooling 때문에 에러가 납니다. -> 추후 수정 부탁드립니다. .__. 
    out_channels = [64, 128, 256, 512, 1280]
    img_resize = (224, 224)
    device = 'cuda'
    lr = 3e-3
    loss = nn.MSELoss()
    epochs = 10
    batch_size = 32
    # batch_size = 128

    transform = A.Compose([
                            A.Resize(img_resize[0], img_resize[0]),
                            ToTensorV2()
                            ])

class encoderModel(nn.Module):
    def __init__(self):
        super(encoderModel, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.global_pool = torch.nn.Identity()
        self.model.classifier = torch.nn.Identity()

    def forward(self, x):
        output = self.model(x)
        return output


class decoderModel(nn.Module):
    def __init__(self):
        super(decoderModel, self).__init__()
        self.model = nn.ModuleList(self.decoderSequential(out_channel, in_channel) for in_channel, out_channel in zip(list(reversed(CFG.in_channels)), list(reversed(CFG.out_channels))))
        # 단순히 out_channel, in_channel을 바꾸고, reversed 함수를 이용하므로서 구현이 가능함.

    def decoderSequential(self, in_channels, out_channels):
        return nn.Sequential(
                                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2)),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(inplace=True),
                            )

    def forward(self, x):
        for model in self.model:
            output = model(x)
            x = output
        return output

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
        # img_name = image.split('\\')[-1]

        image = cv2.imread(image, cv2.IMREAD_COLOR)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            pass

        # label을 지정하면 안될듯 ?
        return image

def train_one_epoch(encoder, decoder, optimizer, dataloader, epoch, loss_fn, device):
    encoder.train()
    decoder.train()

    dataset_size = 0
    running_loss = 0


    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data.to(device, dtype=torch.float)

        batch_size = images.size(0)

        optimizer.zero_grad()
        encode_outputs = encoder(images)
        outputs = decoder(encode_outputs)

        loss = loss_fn(outputs, images)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        train_loss = running_loss/dataset_size
        
        bar.set_postfix(trainEpoch=epoch, trainLoss=train_loss)


def val_one_epoch(encoder, decoder, dataloader, epoch, loss_fn, device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        dataset_size = 0
        running_loss = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for step, data in bar:
            
            images = data.to(device, dtype=torch.float)

            batch_size = images.size(0)

            encode_outputs = encoder(images)
            outputs = decoder(encode_outputs)

            loss = loss_fn(outputs, images)

            running_loss += loss.item()*batch_size
            dataset_size += batch_size
            val_loss = running_loss/dataset_size

            bar.set_postfix(valEpoch=epoch, valLoss=val_loss)
        
        val_loss_arr.append(val_loss)
        if val_loss < min(val_loss_arr[:-1]):
            torch.save(encoder.state_dict(), CFG.weightsavePath+"encoder_model.pt")
            torch.save(decoder.state_dict(), CFG.weightsavePath+"decoder_model.pt")

if __name__ == "__main__":

    fullDataset = ImgDataset(CFG.trainPath, CFG.transform)
    train_size, val_size = 30000, 7500
    # 0.8:0.2 입니다.
    trainDataset, valDataset = random_split(fullDataset, [train_size, val_size])

    trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=CFG.batch_size)
    valDataloader = DataLoader(valDataset, shuffle=False, batch_size=CFG.batch_size)

    encodeModel = encoderModel().to(CFG.device)
    decodeModel = decoderModel().to(CFG.device)

    autoencoder_params = list(encodeModel.parameters()) + list(decodeModel.parameters())

    optimizer = optim.Adam(autoencoder_params, lr=CFG.lr)

    val_loss_arr = [1e+10]

    for epoch in range(1, CFG.epochs+1):
        train_one_epoch(encodeModel, decodeModel, optimizer, trainDataloader, epoch, CFG.loss, CFG.device)
        val_one_epoch(encodeModel, decodeModel, valDataloader, epoch, CFG.loss, CFG.device)

    # embedding_shape = (1, 256, 16, 16)
    # 이거 어려울거 없이 그냥 embedding_dimension임 -> 사이즈 변경시 변경이 필요할 수 있음 -> 이거 쓰레기값으로 들어가는거라서
    # 딱히 상관은 없음 그러나 (1, c, h, w) -> 이런식으로 들어가네용

    
    # https://velog.io/@ppyooy336/Numpy%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-KNN-%EA%B5%AC%ED%98%84
    # 여기참조하고 https://github.com/public-ai/dl-lecture/blob/master/ml-homework/1_Numpy%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20-%20KNN%20%EB%B6%84%EB%A5%98%EA%B8%B0.ipynb
    # 이사람이 배운곳
