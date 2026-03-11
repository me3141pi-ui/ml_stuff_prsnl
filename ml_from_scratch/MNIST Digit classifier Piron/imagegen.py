import Piron
import numpy as np
import matplotlib.pyplot as plt

# Load your custom model
model = Piron.piron.load_model('mnist2.pkl')


def loss(v1, v2):
    # Mean Squared Error logic from your loss.py
    return np.sum((v1 - v2) ** 2)


dig = 0
ideal = np.zeros((10, 1))
ideal[dig][0] = 1
iterations = 700  # Reduced because we'll be more efficient

# Initialize random noise [0, 1] to match your rectifier logic
init = np.random.rand(784, 1) * 0
init = init.reshape((28,28))
init[7:21,7:21] = np.ones((14,14))*100
init = init.reshape((784,1))

for i in range(iterations):
    # We'll store the direction for all pixels here
    change = np.zeros((784, 1))

    # Current prediction and loss for the baseline image
    p_base = model.output(init)
    l_base = loss(ideal, p_base)

    # Instead of deepcopying, we perturb one pixel at a time
    epsilon = 0.001
    for j in range(784):
        original_val = init[j, 0]

        # Test "up"
        init[j, 0] = original_val + epsilon
        l_up = loss(ideal, model.output(init))

        # Test "down"
        init[j, 0] = original_val - epsilon
        l_down = loss(ideal, model.output(init))

        # Revert the pixel to original before moving to next
        init[j, 0] = original_val

        # Determine direction
        if l_up < l_base:
            change[j] = 1
        elif l_down < l_base:
            change[j] = -1

    # Update the image and CLIP to [0, 1] range
    init = np.clip(init + change * 0.0009, 0, 1)


    if i % 10 == 0:
        print(f"Iteration {i} | Current Loss: {l_base}")

plt.imshow(init.reshape((28, 28)), cmap='gray')
print(model.output(init))
plt.title(f"Generated Digit {dig}")
plt.show()