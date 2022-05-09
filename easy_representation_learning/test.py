from glob import glob
from turtle import forward
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class CFG:
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [16, 32, 64, 128, 256]
    pass

class encoderModel(nn.Module):
    def __init__(self):
        super(encoderModel, self).__init__()
        self.model = nn.Sequential(self.encoderSequential(in_channel, out_channel) for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels))
    
    def encoderSequential(in_channels, out_channels):
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
        self.model = nn.Sequential(self.decoderSequential(out_channel, in_channel) for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels))
        # 단순히 out_channel, in_channel을 바꿈으로서 구현이 가능함.

    def decoderSequential(in_channels, out_channels):
        return nn.Sequential(
                                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2)),
                                nn.ReLU(inplace=True)
                            )

    def forward(self, x):
        return self.model(x)

class ImgDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        super(ImgDataset, self).__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            pass

        return image, label

model = encoderModel()

print(model)

# train_img_path = './dataset/train/'
# val_img_path = './dataset/val/'

# train_imgList = sorted(glob())
# val_imgList = 