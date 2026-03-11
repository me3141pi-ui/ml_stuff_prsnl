import numpy

import Piron
import random
import numpy as np
from sklearn.datasets import fetch_openml
import pickle
print("Fetching MNIST data (this may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# Normalize features (0-255 -> 0.0-1.0)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# One-Hot Encode Labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
Y = []
for label in y:
    vec = [0] * 10
    vec[label] = 1
    Y.append(vec)

# Zip into a list of tuples [(x, y), ...]
data = list(zip(X, Y))
random.shuffle(data)

# Split into Train/Test (Optional, but good practice)
train_size = 70000
train_data = data[:train_size]
test_data = data[train_size:train_size + 1000]

print(f"Data loaded. Training on {len(train_data)} samples.")
def training_and_saving():
    # --- 1. Load and Preprocess Data ---


    # --- 2. Build the Model ---
    model = Piron.piron(loss='mse')

    # Input: 784 (28x28 pixels) -> Hidden: 128 -> ReLU
    model.add(Piron.pillar(784, 128, 'relu'))

    # Hidden: 128 -> Hidden: 64 -> ReLU
    # morph=True ensures dimensions align with previous layer automatically
    model.add(Piron.pillar(128, 64, 'relu'), morph=True)

    # Hidden: 64 -> Output: 10 (Digits 0-9) -> Softmax
    model.add(Piron.pillar(64, 10, 'softmax'))

    # --- 3. Setup Optimizer & Scheduler ---
    # Load Data into model
    model.load_data(train_data)
    model.rectify_data()  # Reshapes inputs to column vectors (N, 1)

    # Scheduler: Decay LR slightly every 1000 steps
    model.load_scheduler(
        Piron.scheduler.step_decay(
            initial_lr=0.00007,
            k=0.0005,
            step_size=20
        )
    )

    # Optimizer: Adam is generally best for MNIST
    model.load_optimizer(
        Piron.downhill.Adam(beta1=0.9, beta2=0.999)
    )

    # --- 4. Train ---
    print("Starting training...")
    history = model.train(
        epochs=10,  # Reduced to 5 for quick testing; increase to 20 for full results
        batch_size=64,
        tolerance_loss=0.05
    )

    # --- 5. Save Predictions ---
    print("Saving predictions to mnist_predictions.txt...")
    with open("mnist_predictions.txt", "w") as f:
        # Running on test_data to see generalization (or use 'data' for all)
        subset_to_test = test_data  # Test first 100 for speed

        for x, y_true in subset_to_test:
            pred = model.output(x)

            expected = np.argmax(y_true)
            predicted = np.argmax(pred)

            f.write(f"expected : {expected} | predicted : {predicted}\n")

    print("Done.")

    model.save_model('mnist2.pkl')
    print("Model saved successfully!")


# training_and_saving()

model = Piron.piron.load_model('mnist2.pkl')
#dig = test_data[80][0]
import PIL.Image as img
digim = img.open('./testing_img/2.jpg')
digim = digim.convert('L')

dig = np.array(digim).flatten()/255

import matplotlib.pyplot as plt
plt.imshow(dig.reshape((28,28)),cmap='gray')
print(np.argmax(model.output(dig)))
plt.show()