import torch
import matplotlib.pyplot as plt
import numpy as np
import piron


def deep_dream_generate(model, target_class, img_size=(32, 32), steps=200, lr=0.1, noise_factor=0.02):
    device = next(model.parameters()).device
    model.eval()

    # 1. Custom Image Size injected here
    image = torch.randn(1, 3, img_size[0], img_size[1], device=device, requires_grad=True)

    for i in range(steps):
        output = model(image)

        # We want to MAXIMIZE this score (Gradient Ascent)
        score = output[0, target_class]

        score.backward()

        # Manually add the gradients to the image
        with torch.no_grad():
            image += lr * image.grad

            # 2. The "Stable Diffusion" Hack: Inject random noise back into the tensor
            # We scale the noise by 'noise_factor' so it doesn't completely destroy the image
            image += noise_factor * torch.randn_like(image)

            # Reset gradients for next step
            image.grad.zero_()

        if i % 20 == 0:
            print(f"Step {i} | Score: {score.item():.4f}")

    # Convert to show
    img_tensor = image.detach().cpu().squeeze()

    # Normalize simply for display
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    plt.imshow(img_tensor.permute(1, 2, 0).numpy())
    plt.title(f"Dreamed Class: {target_class}")
    plt.show()

    return image


n = 0
cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Targeting: {cifar10_classes[n]}")

# Example Run: Generating a 64x64 image with a noise factor of 0.05
deep_dream_generate(
    piron.piron.load_model('model_config_files/cifar10/cifar3.4.pft'),
    target_class=n,
    img_size=(32,32),  # Customize your resolution here
    steps=400,
    lr=0.001,
    noise_factor=0.10  # Adjust how chaotic the diffusion is
)