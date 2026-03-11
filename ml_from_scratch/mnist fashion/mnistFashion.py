import numpy as np
import Piron
import random
from sklearn.datasets import fetch_openml
import pickle
import PIL.Image as img
import matplotlib.pyplot as plt

print("Fetching Fashion MNIST data...")
mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

Y = []
for label in y:
    vec = [0] * 10
    vec[label] = 1
    Y.append(vec)

data = list(zip(X, Y))

train_size = 60000
train_data = data[:train_size]
test_data = data[train_size:]

print(f"Data loaded. Training on {len(train_data)} samples.")

def training_and_saving():
    model = Piron.piron(loss='mse')

    model.add(Piron.pillar(784, 512, 'relu'))
    model.add(Piron.pillar(512, 256, 'relu'))
    model.add(Piron.pillar(256, 128, 'relu'))
    model.add(Piron.pillar(128, 64, 'relu'))
    model.add(Piron.pillar(64, 10, 'softmax'), morph=True)

    model.load_data(train_data)
    model.rectify_data()

    model.load_scheduler(
        Piron.scheduler.step_decay(
            initial_lr=0.00001,
            k=0.0005,
            step_size=3
        )
    )

    model.load_optimizer(
        Piron.downhill.Adam(beta1=0.9, beta2=0.9)
    )

    print("Starting training...")
    history = model.train(
        epochs=10,
        batch_size=50,
        tolerance_loss=0.05
    )

    with open("fashion_predictions.txt", "w") as f:
        subset_to_test = test_data[:100]

        for x, y_true in subset_to_test:
            pred = model.output(x)

            expected = np.argmax(y_true)
            predicted = np.argmax(pred)

            f.write(f"expected : {expected} | predicted : {predicted}\n")

    print("Done.")

    model.save_model('fashion_mnist.pkl')
    print("Model saved successfully!")

def training_and_saving2():
    model = Piron.piron.load_model('fashion_mnist2.pkl')

    model.load_data(train_data)
    model.rectify_data()

    model.load_scheduler(
        Piron.scheduler.step_decay(
            initial_lr=0.001,
            k=0.0005,
            step_size=2
        )
    )

    model.load_optimizer(
        Piron.downhill.Adam(beta1=0.9, beta2=0.99)
    )

    print("Starting training...")
    history = model.train(
        epochs=10,
        batch_size=50,
        tolerance_loss=0.05
    )

    with open("fashion_predictions3.txt", "w") as f:
        subset_to_test = test_data[:100]

        for x, y_true in subset_to_test:
            pred = model.output(x)

            expected = np.argmax(y_true)
            predicted = np.argmax(pred)

            f.write(f"expected : {expected} | predicted : {predicted}\n")

    print("Done.")

    model.save_model('fashion_mnist3.pkl')
    print("Model saved successfully!")


#training_and_saving2()
#
model = Piron.piron.load_model('fashion_mnist2.pkl')

correct = 0
total = 0
for point in test_data:
    dig = point[0];y = point[1]

    #plt.imshow(dig.reshape((28, 28)), cmap='jet')
    y_act = np.argmax(y);y_pred = np.argmax(model.output(dig))
    if y_act == y_pred:
        correct += 1
    total += 1
    #plt.show()
print(correct/total)
print(correct,total)