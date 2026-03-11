import numpy as np
import piron
import torch
from torchvision.transforms import ToTensor
import torch.nn as nn
import torchvision
from torchvision import transforms
print('Loading Data...')

train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Mirror the image (p=0.5)
    transforms.RandomCrop(32, padding=4), # Shift the 9image slightly
    transforms.ToTensor(),
    piron.AddGaussianNoise(0,0.01)
])

# Use this 'train_tfms' when loading datasets.CIFAR10

training_data = torchvision.datasets.CIFAR10(
    root = 'data',
    train = True,
    download=True,
    transform=train_tfms
)

dataloader = torch.utils.data.DataLoader(training_data,batch_size=32,shuffle=True)

print('Completed data loading...')

def training_new(filename):
    x = piron.piron()

    short1 = piron.customResnet(path_layers=[
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
    nn.BatchNorm2d(32)
    ], shorting_layers=[
        nn.Conv2d(3,32,1,1,bias=False),
        nn.BatchNorm2d(32)
    ])

    x.add_layer(short1)
    x.add_layer(nn.ReLU())
    short2 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64)
    ],shorting_layers=[
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(64)
    ])
    x.add_layer(short2)
    x.add_layer(nn.ReLU())
    x.add_layer(nn.MaxPool2d(kernel_size=2))

    short3 = piron.customResnet(path_layers=[
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128)
    ],shorting_layers=[
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1,bias=False)
    ])

    x.add_layer(short3)
    x.add_layer(nn.ReLU())

    short4 = piron.customResnet(
        path_layers=[
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128)
        ],
        shorting_layers=[
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1,bias=False)
        ]
    )
    x.add_layer(short4)
    x.add_layer(nn.ReLU())
    x.add_layer((nn.MaxPool2d(kernel_size=2)))

    x.add_layer(nn.Flatten())

    x.add_layer(nn.Linear(8192,4096))
    x.add_layer(nn.ReLU())
    x.add_layer(nn.Dropout(0.2))
    x.add_layer(nn.Linear(4096,512))
    x.add_layer(nn.ReLU())
    x.add_layer(nn.Dropout(0.20))
    x.add_layer(nn.Linear(512,10))

    x.initialise_sequential()
    x.load_loss(nn.CrossEntropyLoss())
    x.load_optimizer(torch.optim.SGD(x.parameters(),lr = 0.005))
    optimizer = torch.optim.SGD(x.parameters(), lr=0.06)
    x.load_optimizer(optimizer)
    x.load_dataloader(dataloader)
    x.epoch_run(epoch = 20,tol_loss=150)
    x.save_model(filename)

def resume_training(filename):
    x = piron.piron.load_model(filename)
    x.load_dataloader(dataloader)
    x.epoch_run(epoch = 5,tol_loss=100)
    x.save_model('cifar3.7.pft')
print('Starting training process ...')
resume_training('model_config_files/cifar10/cifar3.6.pft')