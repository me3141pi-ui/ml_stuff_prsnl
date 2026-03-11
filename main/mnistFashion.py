
import piron
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
print('Loading data')
training_data = datasets.FashionMNIST(
    root = 'data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data',
    transform=ToTensor(),
    download=True,
    train = False
)
train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=1)


x  = piron.piron()
x.add_layer(nn.Flatten())
x.add_layer(nn.Linear(784,512))
x.add_layer(nn.ReLU())
x.add_layer(nn.Linear(512,256))
x.add_layer(nn.ReLU())
x.add_layer(nn.Linear(256,128))
x.add_layer(nn.ReLU())
x.add_layer(nn.Linear(128,64))
x.add_layer(nn.ReLU())
x.add_layer(nn.Linear(64,10))

x.initialise_sequential()

x.load_dataloader(train_dataloader)

x.load_loss(nn.CrossEntropyLoss())

opt = torch.optim.SGD(x.parameters(),lr = 1e-3)
x.load_optimizer(opt)

x.epoch_run(epoch = 50)

x.save_model('model4.pft')

# print('Loading model')
# model = piron.piron.load_model('model3.pft')
# model.load_dataloader(train_dataloader)
# model.epoch_run(epoch = 10)
# model.save_model('model3.pft')
# correct = total = 0
# for x,y in test_dataloader:
#     pred = model.predict(x)
#
#     res = pred.argmax(1);
#
#     #print(res,y)
#     if res.item() == y.item():
#         correct += 1
#     total += 1
#
# print(correct/total)
# print(correct , total)
