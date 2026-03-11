import numpy as np
import piron
import torch
from torchvision.transforms import ToTensor
import torch.nn as nn
import torchvision

model = piron.piron.load_model('model_config_files/cifar10/cifar3.5.pft')

testing_data = torchvision.datasets.CIFAR10(
    root = 'data',
    train = False,
    download=True,
    transform=ToTensor()
)
correct = total = 0
for x,y in torch.utils.data.DataLoader(testing_data,batch_size=1):
    pred = model.predict(x)
    pred = pred.argmax(1)
    if pred.item() == y.item():
        correct += 1
    total += 1

print(correct/total)
print(correct,total)