import torch
from torchvision import transforms
from PIL import Image
import piron

cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def predict_image(model, image_path):
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = logits.argmax(1).item()

    return logits
model = piron.piron.load_model('model_config_files/cifar10/cifar3.5.pft')
lgit = predict_image(model,'test_data_me/frog.jpeg')
print(lgit)
print(cifar10_classes[lgit.argmax(1).item()])