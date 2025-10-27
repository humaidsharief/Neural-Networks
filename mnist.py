from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range (5):
    plt.subplot(1, 5, i +1) # create a grid of 1 row and 5 columns
    plt.imshow(x_train[i], cmap="gray") # make the grid in greyscale
    plt.title(f"Number: {y_train[i]}") # name each tile correctly
    plt.axis("off") # remove the axes for convenience
plt.show()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
model = Sequential([
    Dense(128, activation="relu", input_shape=(784, )),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=5, batch_size=32,
    validation_data=(x_test, y_test)
)

model.save("mnist_model.h5")