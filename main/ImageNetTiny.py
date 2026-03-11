import os

import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageFolder
import piron


def correcting_val():
    import shutil
    from pathlib import Path

    with open('data/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f.readlines():

            content = line.split()
            imgFile = content[0]
            clas = content[1]
            print(imgFile,clas)
            start_dest = 'data/tiny-imagenet-200/val/images/' + imgFile
            end_dest = 'data/tiny-imagenet-200/validation/' + clas + '/'

            folder_path = Path(end_dest)
            if folder_path.is_dir():
                shutil.copy2(start_dest,end_dest)
            else:
                os.mkdir(end_dest)
                shutil.copy2(start_dest, end_dest)




def training_and_saving():

    rndmTfms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomCrop(64,padding=8),
        torchvision.transforms.ToTensor(),
        piron.AddGaussianNoise(std=0.01,mean=0)
   ])
    #note that training data has images of the form
    print('Loading Data ...')
    train_data = ImageFolder('data/tiny-imagenet-200/train',transform=rndmTfms)
    dataloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    print('Loaded Data ...')

    import torch.nn as nn
    print('Starting Training')

    model = piron.piron()

    model.add_layer(
        nn.Conv2d(3,64,7,stride=2,padding=3)
    )
    model.add_layer((nn.MaxPool2d(kernel_size=2)))

    short1 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64)
    ],
    shorting_layers=None)
    model.add_layer(short1)
    model.add_layer(nn.ReLU())

    short2 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64)
    ],
    shorting_layers=None)
    model.add_layer(short2)
    model.add_layer(nn.ReLU())

    short3 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride = 2,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride = 1,padding=1),
        nn.BatchNorm2d(128)
    ],shorting_layers=[nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=2,bias=False)])
    model.add_layer(short3)
    model.add_layer(nn.ReLU())

    short4 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128)
    ],shorting_layers=None)
    model.add_layer(short4)
    model.add_layer(nn.ReLU())

    short5 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride = 2,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
        nn.BatchNorm2d(256)
    ],shorting_layers=[nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=2,bias=False)])
    model.add_layer(short5)
    model.add_layer(nn.ReLU())

    short6 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256)
    ], shorting_layers=None)
    model.add_layer(short6)
    model.add_layer(nn.ReLU())

    short7 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride = 2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512)
    ], shorting_layers=[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1,stride = 2,  bias=False)])
    model.add_layer(short7)
    model.add_layer(nn.ReLU())

    short8 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512)
    ], shorting_layers=None)
    model.add_layer(short8)
    model.add_layer(nn.ReLU())

    model.add_layer(nn.AdaptiveAvgPool2d((1,1)))
    model.add_layer(nn.Flatten())

    model.add_layer(nn.Linear(512,200))

    model.initialise_sequential()

    model.load_dataloader(dataloader)
    model.load_optimizer(torch.optim.SGD(model.parameters(),lr = 0.01))
    model.load_loss(nn.CrossEntropyLoss())

    model.epoch_run(epoch=10)
    model.save_model('ImgResNet1.0.ptf')

training_and_saving()



