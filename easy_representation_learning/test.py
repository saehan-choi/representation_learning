import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from glob import glob
import cv2
import numpy as np

class CFG:
    # imagePath
    # 이거 jpg말고도 될수있게 한번ㄱ..ㄱ;
    weightsavePath = './weights/'
    embeddingsavePath = './embedding/'

    trainPath = glob('./dataset/train/*.jpg')
    valPath = glob('./dataset/val/*.jpg')

    # 이것만 있으면 image의 사이즈를 줄였다가, 또 늘릴수 있음 같은크기로 ( list(reversed()) 를 통해서 )
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [16, 32, 64, 128, 256]

    img_resize = (96, 96)

    device = 'cuda'
    
    lr = 3e-4
    loss = nn.MSELoss()
    epochs = 5
    batch_size = 128

    transform = A.Compose([
                            A.Resize(img_resize[0], img_resize[0]),
                            ToTensorV2()
                            ])

class encoderModel(nn.Module):
    def __init__(self):
        super(encoderModel, self).__init__()
        self.model = nn.ModuleList(self.encoderSequential(in_channel, out_channel) for in_channel, out_channel in zip(CFG.in_channels, CFG.out_channels))
    def encoderSequential(self, in_channels, out_channels):
        return nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)),
                                nn.ReLU(inplace=True),
                                # inplace True하면 본래 input값을 수정하겠단 말임 -> 따라서 메모리 사용량이 좋아짐.
                                nn.MaxPool2d((2,2))
                            )
    def forward(self, x):
        for model in self.model:
            output = model(x)
            x = output
        return output


class decoderModel(nn.Module):
    def __init__(self):
        super(decoderModel, self).__init__()
        self.model = nn.ModuleList(self.decoderSequential(out_channel, in_channel) for in_channel, out_channel in zip(list(reversed(CFG.in_channels)), list(reversed(CFG.out_channels))))
        # 단순히 out_channel, in_channel을 바꾸고, reversed 함수를 이용하므로서 구현이 가능함.

    def decoderSequential(self, in_channels, out_channels):
        return nn.Sequential(
                                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2)),
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
        # print(f'output:{encode_outputs.size()}')
        # print(f'reshape:{encode_outputs.reshape((encode_outputs.size()[0], -1)).size()}')
        # fla = torch.flatten(encode_outputs, start_dim=1).size()
        # print(f'flatten:{fla}')

        outputs = decoder(encode_outputs)

        loss = loss_fn(outputs, images)
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad -> loss.backward -> optimizer.step 이 순서대로 하면됨.

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

# 이 밑에 두개함수는 학습 잘되면, torch load해서 가져오고, 테스트 하도록하기
# 그게 더 편함.
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
            # 이거 cpu 빼보고 해보기 --> torch.randn에서 cuda를 안사용해서 X
            # keep adding these outputs to embeddings.  0번째 차원에서 concatenation
            embedding = torch.cat((embedding, encode_outputs), 0)

    return embedding

def compute_similar_iamges(encoder, image, num_images, embedding, device):
    """
    image: Image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """
    image_tensor = CFG.transform(image)
    # 이거 에러날수도 있음 확인
    image_tensor = image_tensor.unsqueeze(0)
    # 0번째 차원에 텐서 dimension 확장 -> torch에 넣어주기 위해

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        # flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
        # 여기서 변경했습니다 에러나면 변경하시오.
    flattend_embedding = torch.flatten(image_embedding, start_dim=1)

# https://velog.io/@ppyooy336/Numpy%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-KNN-%EA%B5%AC%ED%98%84
# 여기참조하고 https://github.com/public-ai/dl-lecture/blob/master/ml-homework/1_Numpy%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%20-%20KNN%20%EB%B6%84%EB%A5%98%EA%B8%B0.ipynb
# 이사람이 배운곳

def k_nearest_neibor(input, dataset, labels, K):
    # k = a-b
    # a = embedding_image  b = search_image
    # root((a[0]-b[0])^2+(a[1]-b[1])^2+(a[2]-b[2])^2)
    # 이런방식으로 진행해야 합니다.

    # 정규화 + 후에 np.sqrt(k^2)
    # 정규화를 하는 대신에 z 표준화를 하는방식이 제일 많이쓰인답니다.
    square = (dataset-input)**2
    distance = np.sqrt(square.sum(axis=1))


if __name__ == "__main__":

    trainDataset = ImgDataset(CFG.trainPath, CFG.transform)
    valDataset = ImgDataset(CFG.valPath, CFG.transform)

    trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=CFG.batch_size)
    valDataloader = DataLoader(valDataset, shuffle=False, batch_size=CFG.batch_size)

    encodeModel = encoderModel().to(CFG.device)
    decodeModel = decoderModel().to(CFG.device)

    autoencoder_params = list(encodeModel.parameters()) + list(decodeModel.parameters())

    optimizer = optim.Adam(autoencoder_params, lr=CFG.lr)

    val_loss_arr = [1e+10]
    # loss값 저장을위해 trash value 한번해줌

    for epoch in range(1, CFG.epochs+1):
        train_one_epoch(encodeModel, decodeModel, optimizer, trainDataloader, epoch, CFG.loss, CFG.device)
        val_one_epoch(encodeModel, decodeModel, valDataloader, epoch, CFG.loss, CFG.device)


    # embedding_shape = (1, 256, 16, 16)
    # 이거 어려울거 없이 그냥 embedding_dimension임 -> 사이즈 변경시 변경이 필요할 수 있음
    embedding_shape = (1, 256, int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))), int(CFG.img_resize[0]/(2**(len(CFG.out_channels)))))

    # 이거 마지막으로 나오는 encoder dimension 인데 쓰레기값으로 들어갈것이므로 ㄱㅊㄱㅊ
    embedding = create_embedding(encodeModel, trainDataloader, embedding_shape, CFG.device)
    
    flattend_embedding = torch.flatten(embedding, start_dim=1).cpu().detach().numpy()

    np.save(CFG.embeddingsavePath+"data_embedding.npy", flattend_embedding)




