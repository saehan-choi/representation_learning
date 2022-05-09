from glob import glob
from turtle import forward
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class CFG:
    # imagePath
    trainPath = './dataset/train/'
    valPath = './dataset/val/'

    # 이것만 있으면 image의 사이즈를 줄였다가, 또 늘릴수 있음 같은크기로 ( list(reversed()) 를 통해서 )
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [16, 32, 64, 128, 256]



class encoderModel(nn.Module):
    def __init__(self):
        super(encoderModel, self).__init__()
        self.layers = []
        self.model = nn.ModuleList(self.encoderSequential(in_channel, out_channel) for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels))

    def encoderSequential(self, in_channels, out_channels):
        return nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)),
                                nn.ReLU(inplace=True),
                                # inplace True하면 본래 input값을 수정하겠단 말임 -> 따라서 메모리 사용량이 좋아짐.
                                nn.MaxPool2d((2,2))
                            )
    def forward(self, x):
        return self.model(x)


class decoderModel(nn.Module):
    def __init__(self):
        super(decoderModel, self).__init__()
        self.layers = []
        self.model = nn.ModuleList(self.decoderSequential(out_channel, in_channel) for in_channel, out_channel in zip(list(reversed(CFG.in_channels)), list(reversed(CFG.out_channels))))
        # 단순히 out_channel, in_channel을 바꾸고, reversed 함수를 이용하므로서 구현이 가능함.

    def decoderSequential(self, in_channels, out_channels):
        return nn.Sequential(
                                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2)),
                                nn.ReLU(inplace=True),
                            )

    def forward(self, x):
        return self.model(x)

class ImgDataset(Dataset):
    def __init__(self, image_list, transform=None):
        # 여기서는 딱히 나눌필요 없어보임 왜냐면, representation learning 을 이용하기 때문
        # label도 필요없네요.
        super(ImgDataset, self).__init__()
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            pass
        
        # label을 지정하면 안될듯 ?
        return image

encodeModel = encoderModel()
decodeModel = decoderModel()

def train_one_epoch(encoder, decoder, optimizer, dataloader, epoch, train_loss_arr, device):
    encoder.train()
    decoder.train()

    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        images = data.to(device, dtype=torch.float)

        batch_size = images.size(0)

        encode_outputs = encoder(images)
        decode_outputs = decoder(encode_outputs)

        loss = nn.CrossEntropyLoss()(decode_outputs, images)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(trainEpoch=epoch, trainLoss=epoch_loss)

    train_loss_arr.append(epoch_loss)

def val_one_epoch(encoder, decoder, optimizer, dataloader, epoch, val_loss_arr, device):
    encoder.eval()
    decoder.eval()

    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        images = data.to(device, dtype=torch.float)

        batch_size = images.size(0)

        encode_outputs = encoder(images)
        decode_outputs = decoder(encode_outputs)

        loss = nn.CrossEntropyLoss()(decode_outputs, images)

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(valEpoch=epoch, valLoss=epoch_loss)
    val_loss_arr.append(epoch_loss)


train_img_path = CFG.trainPath
val_img_path = CFG.valPath

print(train_img_path)

# train_imgList = sorted(glob())
# val_imgList = 